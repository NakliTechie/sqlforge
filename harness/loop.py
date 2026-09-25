"""Harness — the writable surface. Talks to the model server, formats observations, decides when the episode ends.
It never touches hidden snapshots; run_sql is a callable handed in by the lockbox.

Two backends, one turn loop:
- "ollama": Ollama /api/chat (laptop; 4-bit weights; cannot batch qwen35 as of 0.30.11)
- "openai": an OpenAI-compatible /v1/chat/completions server, i.e. vLLM with --enable-auto-tool-choice
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 (GPU; bf16; batches concurrent episodes)
Each adapter returns a normalised turn {thinking, content, calls:[{id, name, args}]} or {malformed}; the turn rule
in harness/parse.py and the history format are shared.
"""
import json
from pathlib import Path

import requests

from harness.nudge import budget_note
from harness.parse import EMPTY_TURN_NOTE, parse_calls, sql_in_answer

HERE = Path(__file__).parent
URLS = {"ollama": "http://localhost:11434/api/chat", "openai": "http://127.0.0.1:8000/v1/chat/completions"}
OBS_ROWS = 20
OBS_CHARS = 1500
NUM_PREDICT = 2048  # per-turn generation cap; a runaway turn ends as a plain text turn instead of a 600 s timeout


class ModelError(Exception):
    pass


def _fmt_obs(cols, rows, err):
    if err:
        return f"ERROR: {err}"
    if not rows:
        return f"columns: {cols}\n(0 rows)"
    head = " | ".join(cols)
    body = "\n".join(" | ".join("NULL" if v is None else str(v) for v in r) for r in rows[:OBS_ROWS])
    more = f"\n... ({len(rows)} rows total, showing {OBS_ROWS})" if len(rows) > OBS_ROWS else f"\n({len(rows)} rows)"
    return (head + "\n" + body + more)[:OBS_CHARS]


def _post(url: str, payload: dict) -> requests.Response:
    r = requests.post(url, json=payload, timeout=600)
    if r.status_code != 200 and not (r.status_code == 500 and "syntax error" in r.text):
        r = requests.post(url, json=payload, timeout=600)  # one retry on server errors
    return r


def _ollama_turn(url, model, msgs, tools, *, seed, temperature, num_ctx, think) -> dict:
    r = _post(url, {"model": model, "messages": msgs, "tools": tools, "stream": False, "think": think,
                    "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx, "num_predict": NUM_PREDICT}})
    if r.status_code == 500 and "syntax error" in r.text:  # Ollama could not parse the model's tool-call XML
        return {"malformed": r.text[:300]}
    if r.status_code != 200:
        raise ModelError(f"ollama {r.status_code}: {r.text[:300]}")
    m = r.json()["message"]
    return {"thinking": m.get("thinking") or "", "content": m.get("content") or "",
            "calls": [{"id": None, "name": c["function"]["name"], "args": c["function"].get("arguments") or {}}
                      for c in (m.get("tool_calls") or [])]}


def _openai_turn(url, model, msgs, tools, *, seed, temperature, num_ctx, think) -> dict:
    r = _post(url, {"model": model, "messages": msgs, "tools": tools, "temperature": temperature, "seed": seed,
                    "max_tokens": NUM_PREDICT, "chat_template_kwargs": {"enable_thinking": think}})
    if r.status_code != 200:
        raise ModelError(f"openai {r.status_code}: {r.text[:300]}")
    m = r.json()["choices"][0]["message"]
    calls = []
    for c in m.get("tool_calls") or []:
        try:
            args = json.loads(c["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append({"id": c.get("id"), "name": c["function"]["name"], "args": args if isinstance(args, dict) else {}})
    return {"thinking": m.get("reasoning_content") or m.get("reasoning") or "", "content": m.get("content") or "",
            "calls": calls}


TURN = {"ollama": _ollama_turn, "openai": _openai_turn}


def _assistant_msg(backend: str, content: str, calls: list[dict]) -> dict:
    """History entry for an assistant turn. Calls parsed from text (harness/parse.py rule 1) are written back as
    structured calls, so the next request's history matches what the harness executed."""
    msg = {"role": "assistant", "content": content}
    if calls:
        if backend == "openai":
            msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for c in calls]
        else:
            msg["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["args"]}} for c in calls]
    return msg


def run_episode(ddl: str, question: str, run_sql, *, model: str, seed: int, max_turns: int, num_ctx: int,
                temperature: float = 0.6, think: bool = True, backend: str = "ollama", url: str | None = None) -> dict:
    system = (HERE / "system.md").read_text().replace("{ddl}", ddl.strip())
    tools = json.loads((HERE / "tools.json").read_text())
    url = url or URLS[backend]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    trace = [{"role": "user", "content": question}]  # forensics copy: keeps thinking, which msgs drops
    turns, n_run_sql, submitted, via = 0, 0, None, None
    while turns < max_turns:
        turns += 1
        try:
            t = TURN[backend](url, model, msgs, tools, seed=seed, temperature=temperature, num_ctx=num_ctx, think=think)
        except (requests.exceptions.RequestException, ModelError) as e:
            detail = str(e) if isinstance(e, ModelError) else f"request error: {type(e).__name__}"
            return {"submitted_sql": None, "via": "error", "turns": turns, "n_run_sql": n_run_sql, "hit_cap": False,
                    "final_text": detail, "trace": trace}
        if "malformed" in t:
            via = "malformed"
            trace.append({"role": "error", "content": t["malformed"]})
            break
        calls = t["calls"] or parse_calls(t["content"]) or parse_calls(t["thinking"])  # harness/parse.py rule 1
        for i, c in enumerate(calls):
            c["id"] = c.get("id") or f"call_{turns}_{i}"
        msgs.append(_assistant_msg(backend, t["content"], calls))
        trace.append({"role": "assistant", "thinking": t["thinking"], "content": t["content"],
                      "tool_calls": [{"name": c["name"], "args": c["args"]} for c in calls]})
        if not calls:
            sql = sql_in_answer(t["content"])                                               # rule 2
            if sql:
                submitted, via = sql, "text"
                break
            msgs.append({"role": "user", "content": EMPTY_TURN_NOTE + budget_note(max_turns - turns)})  # rule 3
            trace.append(msgs[-1])
            continue
        done, results = False, []
        for c in calls:
            fn, args = c["name"], c["args"]
            sql = args.get("sql", "") if isinstance(args, dict) else ""
            if fn == "submit":
                submitted, via, done = sql, "tool", True
                break
            if fn == "run_sql":
                n_run_sql += 1
                cols, rows, err = run_sql(sql)
                results.append((c, _fmt_obs(cols, rows, err)))
            else:
                results.append((c, f"ERROR: unknown tool {fn}"))
        if done:
            break
        note = budget_note(max_turns - turns)
        if backend == "openai":  # one tool message per call id; the budget note rides on the last one
            for k, (c, obs) in enumerate(results):
                msgs.append({"role": "tool", "tool_call_id": c["id"],
                             "content": obs + (note if k == len(results) - 1 else "")})
                trace.append(msgs[-1])
        else:
            msgs.append({"role": "tool", "content": "\n\n".join(obs for _, obs in results) + note})
            trace.append(msgs[-1])
    last = next((m.get("content") or "" for m in reversed(msgs) if m["role"] == "assistant"), "")
    return {"submitted_sql": submitted, "via": via, "turns": turns, "n_run_sql": n_run_sql,
            "hit_cap": turns >= max_turns and submitted is None, "final_text": last[:600], "trace": trace}
