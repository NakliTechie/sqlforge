"""Harness v0 — the writable surface. Talks to Ollama, formats observations, decides when the
episode ends. It never touches hidden snapshots; run_sql is a callable handed in by the lockbox."""
import json
import re
from pathlib import Path

import requests

from harness.nudge import budget_note

HERE = Path(__file__).parent
OLLAMA = "http://localhost:11434/api/chat"
OBS_ROWS = 20
OBS_CHARS = 1500
NUM_PREDICT = 2048  # per-turn generation cap; a runaway turn ends as a plain text turn instead of a 600 s timeout


def _fmt_obs(cols, rows, err):
    if err:
        return f"ERROR: {err}"
    if not rows:
        return f"columns: {cols}\n(0 rows)"
    head = " | ".join(cols)
    body = "\n".join(" | ".join("NULL" if v is None else str(v) for v in r) for r in rows[:OBS_ROWS])
    more = f"\n... ({len(rows)} rows total, showing {OBS_ROWS})" if len(rows) > OBS_ROWS else f"\n({len(rows)} rows)"
    return (head + "\n" + body + more)[:OBS_CHARS]


def _sql_from_text(text: str):
    m = re.findall(r"```sql\s*(.*?)```", text, flags=re.S | re.I)
    return m[-1].strip() if m else None


def run_episode(ddl: str, question: str, run_sql, *, model: str, seed: int, max_turns: int, num_ctx: int,
                temperature: float = 0.6) -> dict:
    system = (HERE / "system.md").read_text().replace("{ddl}", ddl.strip())
    tools = json.loads((HERE / "tools.json").read_text())
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    turns, n_run_sql, submitted, via = 0, 0, None, None
    while turns < max_turns:
        turns += 1
        try:
            r = requests.post(OLLAMA, json={"model": model, "messages": msgs, "tools": tools, "stream": False,
                                            "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx, "num_predict": NUM_PREDICT}},
                              timeout=600)
        except requests.exceptions.RequestException as e:
            return {"submitted_sql": None, "via": "error", "turns": turns, "n_run_sql": n_run_sql, "hit_cap": False,
                    "final_text": f"request error: {type(e).__name__}"}
        if r.status_code == 500 and "syntax error" in r.text:
            # Ollama could not parse the model's tool call (malformed XML): a model failure, not infrastructure.
            via, msgs = "malformed", msgs + [{"role": "assistant", "content": r.text[:300]}]
            break
        if r.status_code != 200:
            r = requests.post(OLLAMA, json={"model": model, "messages": msgs, "tools": tools, "stream": False,
                                            "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx, "num_predict": NUM_PREDICT}},
                              timeout=600)
            if r.status_code != 200:
                return {"submitted_sql": None, "via": "error", "turns": turns, "n_run_sql": n_run_sql, "hit_cap": False,
                        "final_text": f"ollama {r.status_code}: {r.text[:300]}"}
        msg = r.json()["message"]
        msgs.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        calls = msg.get("tool_calls") or []
        if not calls:
            submitted, via = _sql_from_text(msg.get("content") or ""), "text"
            break
        done, obs_parts = False, []
        for c in calls:
            fn, args = c["function"]["name"], c["function"].get("arguments") or {}
            sql = args.get("sql", "") if isinstance(args, dict) else ""
            if fn == "submit":
                submitted, via, done = sql, "tool", True
                break
            if fn == "run_sql":
                n_run_sql += 1
                cols, rows, err = run_sql(sql)
                obs_parts.append(_fmt_obs(cols, rows, err))
            else:
                obs_parts.append(f"ERROR: unknown tool {fn}")
        if done:
            break
        msgs.append({"role": "tool", "content": "\n\n".join(obs_parts) + budget_note(max_turns - turns)})
    last = next((m.get("content") or "" for m in reversed(msgs) if m["role"] == "assistant"), "")
    return {"submitted_sql": submitted, "via": via, "turns": turns, "n_run_sql": n_run_sql,
            "hit_cap": turns >= max_turns and submitted is None, "final_text": last[:600]}
