"""Trainer self-check. Usage: uv run python -m train.selfcheck [--model Qwen/Qwen3.5-0.8B] [--device cpu]

1. shaped_reward: binary pass, length penalty only on passes, no partial credit.
2. budget_note: silent above the nudge window, "Last turn" at the end.
3. sequence_logprobs (logits_to_keep + checkpointed chunks) matches the full-logits reference in value and in
   LoRA gradients, across several chunk boundaries. Runs in fp32 on CPU by default so it never competes with a
   measurement for the GPU.
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from harness.nudge import NUDGE_TURNS, budget_note
from train import grpo


def reference_logprobs(model, ids, mask):
    logits = model(input_ids=ids[:-1].unsqueeze(0)).logits[0].float()
    lp = torch.gather(F.log_softmax(logits, dim=-1), 1, ids[1:].unsqueeze(1)).squeeze(1)
    m = mask[1:].float()
    return (lp * m).sum() / m.sum()


def check_reward():
    r = grpo.shaped_reward
    assert r(False, True, 10, 6000, 0.1) == 0.0            # submitted, wrong
    assert r(False, False, 10, 6000, 0.1) == 0.0           # never submitted, default: same as wrong
    assert r(False, False, 10, 6000, 0.1, -1.0) == -1.0    # never submitted, SkyRL-style penalty
    assert r(True, True, 100, 6000, 0.1) == 1.0
    assert r(True, True, 12000, 6000, 0.0) == 1.0
    assert abs(r(True, True, 12000, 6000, 0.1) - (1 - 0.1 * 0.6931)) < 1e-3
    oc = grpo.outcome_class
    assert grpo.group_advantages([oc(x) for x in (0.0, 0.0, 0.0, 0.0)]) is None          # all wrong → dropped
    assert grpo.group_advantages([oc(x) for x in (-1.0, 0.0, -1.0, 0.0)]) is not None    # no-submit vs wrong → kept
    assert grpo.group_advantages([oc(x) for x in (1.0, 0.93, 1.0, 0.8)]) is None         # all pass, length only → dropped
    print("OK shaped_reward + outcome_class")


def check_nudge():
    assert budget_note(NUDGE_TURNS + 1) == ""
    assert budget_note(NUDGE_TURNS).startswith(f"\n\n[budget] {NUDGE_TURNS} turns left")
    assert "Last turn" in budget_note(1) and "Last turn" in budget_note(0)
    print("OK budget_note")


def check_turn_rule(model_name: str):
    """harness/parse.py rule, exercised through BOTH harnesses on the same scripted turns:
    t1 thinking-only with SQL in the thinking → not a submission, harness note, episode continues;
    t2 tool call written inside the thinking → executed; t3 submit → gold passes."""
    import json as _json
    from unittest import mock

    import harness.loop as L
    from harness import parse
    from lab import schemas, tasks, verify
    from train.rollout import run_episode as hf_episode, tokenize_trajectory
    from transformers import AutoTokenizer

    assert parse.split_thinking("a ```sql\nX\n``` b</think>\n\nans") == ("a ```sql\nX\n``` b", "ans")
    assert parse.split_thinking("cut mid-thought") == ("cut mid-thought", "")
    assert parse.sql_in_answer("no sql here") is None and parse.sql_in_answer("```sql\nSELECT 2\n```") == "SELECT 2"
    assert parse.parse_calls("x<tool_call>\n<function=run_sql>\n<parameter=sql>\nSELECT 1\n</parameter>\n</function>\n</tool_call>") \
        == [{"name": "run_sql", "args": {"sql": "SELECT 1"}}]

    t = tasks.TASKS[0]
    gold = t["gold"][0]
    con = schemas.build(t["schema"], tasks.VISIBLE_SEED)
    def run_sql(q):
        try:
            c, r = verify.execute(con, q); return c, r, None
        except verify.ExecError as e:
            return [], [], str(e)
    xml_call = "<tool_call>\n<function=run_sql>\n<parameter=sql>\nSELECT 1\n</parameter>\n</function>\n</tool_call>"

    # Measurement harness, both backends, same three turns
    class R:
        status_code, text = 200, ""
        def __init__(self, body): self.body = body
        def json(self): return self.body
    ollama_script = [
        {"message": {"role": "assistant", "thinking": "plan:\n```sql\nSELECT 42\n```\nlet me test first", "content": ""}},
        {"message": {"role": "assistant", "thinking": "run it " + xml_call, "content": ""}},
        {"message": {"role": "assistant", "thinking": "done", "content": "",
                     "tool_calls": [{"function": {"name": "submit", "arguments": {"sql": gold}}}]}}]
    openai_script = [
        {"choices": [{"message": {"role": "assistant", "reasoning_content": "plan:\n```sql\nSELECT 42\n```\nlet me test", "content": None}}]},
        {"choices": [{"message": {"role": "assistant", "reasoning_content": "run it " + xml_call, "content": ""}}]},
        {"choices": [{"message": {"role": "assistant", "reasoning_content": "done", "content": None, "tool_calls": [
            {"id": "chatcmpl-tool-1", "type": "function", "function": {"name": "submit", "arguments": _json.dumps({"sql": gold})}}]}}]}]
    for backend, script in (("ollama", ollama_script), ("openai", openai_script)):
        it, sent = iter(script), []
        def post(url, json=None, timeout=None):
            sent.append(_json.loads(_json.dumps(json)))
            return R(next(it))
        with mock.patch.object(L.requests, "post", side_effect=post):
            ep = L.run_episode(t["ddl"], t["q"], run_sql, model="m", seed=1, max_turns=10, num_ctx=4096, backend=backend)
        roles = [m["role"] for m in ep["trace"]]
        assert ep["submitted_sql"] == gold and ep["via"] == "tool" and ep["turns"] == 3 and ep["n_run_sql"] == 1, (backend, ep)
        assert roles == ["user", "assistant", "user", "assistant", "tool", "assistant"], (backend, roles)
        assert ep["trace"][2]["content"].startswith(parse.EMPTY_TURN_NOTE), backend
        hist = sent[-1]["messages"]            # what the 3rd request carried
        a2, tool = hist[-2], hist[-1]           # turn-2 assistant (call parsed from thinking) and its observation
        assert a2["role"] == "assistant" and a2["tool_calls"], (backend, a2)
        if backend == "openai":
            assert tool["role"] == "tool" and tool["tool_call_id"] == a2["tool_calls"][0]["id"], (backend, tool)
            assert "enable_thinking" in sent[0]["chat_template_kwargs"] and sent[0]["max_tokens"] == L.NUM_PREDICT

    # HF training harness, same turns as raw Qwen text (prompt ends in '<think>\n')
    tok = AutoTokenizer.from_pretrained(model_name)
    submit_xml = "<tool_call>\n<function=submit>\n<parameter=sql>\n" + gold + "\n</parameter>\n</function>\n</tool_call>"
    raw = iter(["plan:\n```sql\nSELECT 42\n```\nlet me test first\n</think>\n\n<|im_end|>",
                "run it " + xml_call + "\n</think>\n\n<|im_end|>",
                "done\n</think>\n\n" + submit_xml + "<|im_end|>"])
    class P:
        def generate(self, prompt): return next(raw)
        def generate_batch(self, prompts, seeds): return [self.generate(x) for x in prompts]
    tools = _json.loads(open("harness/tools.json").read())
    tr = hf_episode(P(), tok, "S", t["q"], tools, run_sql, max_turns=10, think=True)
    assert tr.submitted_sql == gold and tr.turns == 3 and not tr.hit_cap, (tr.submitted_sql, tr.turns)
    gen = [txt for txt, g in tr.segments if g]
    assert len(gen) == 3 and parse.EMPTY_TURN_NOTE in tr.segments[2][0]
    ids, mask = tokenize_trajectory(tok, tr)
    assert tok.decode(ids[mask.bool()]) == "".join(gen)
    print("OK turn rule (loop.py ollama + openai and rollout.py agree: note, thinking-call, submit)")


def check_batched_rollout(model_name: str):
    """run_episodes (lockstep batch) must reproduce run_episode (one at a time) exactly, with episodes finishing on
    different turns: A submits on turn 1, B runs SQL then submits on turn 2, C sends an empty turn then caps at 3."""
    import json as _json
    from lab import schemas, tasks, verify
    from train.rollout import Episode, run_episode, run_episodes
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tools = _json.loads(open("harness/tools.json").read())
    t = tasks.TASKS[0]
    con = schemas.build(t["schema"], tasks.VISIBLE_SEED)
    def run_sql(q):
        try:
            c, r = verify.execute(con, q); return c, r, None
        except verify.ExecError as e:
            return [], [], str(e)
    call = lambda n, q: f"x</think>\n\n<tool_call>\n<function={n}>\n<parameter=sql>\n{q}\n</parameter>\n</function>\n</tool_call><|im_end|>"
    scripts = {"QA": [call("submit", "SELECT 1")],
               "QB": [call("run_sql", "SELECT 2"), call("submit", "SELECT 3")],
               "QC": ["just thinking</think>\n\n<|im_end|>", call("run_sql", "SELECT 4"), call("run_sql", "SELECT 5")]}
    class P:
        def __init__(self): self.calls, self.pos = 0, {}
        def generate_batch(self, prompts, seeds):
            self.calls += 1
            out = []
            for pr in prompts:
                q = next(k for k in scripts if f"\n{k}<|im_end|>" in pr)
                i = self.pos.get(q, 0); self.pos[q] = i + 1
                out.append(scripts[q][i])
            return out
        def generate(self, prompt): return self.generate_batch([prompt], [None])[0]
    mk = lambda q: Episode(tok, "S", q, tools, run_sql, max_turns=3, think=True)
    batched = run_episodes(pb := P(), [mk(q) for q in scripts])
    single = [run_episode(P(), tok, "S", q, tools, run_sql, max_turns=3, think=True) for q in scripts]
    for b, s in zip(batched, single):
        assert (b.submitted_sql, b.turns, b.hit_cap, b.segments) == (s.submitted_sql, s.turns, s.hit_cap, s.segments)
    assert [b.turns for b in batched] == [1, 2, 3] and batched[2].hit_cap and pb.calls == 3, [b.turns for b in batched]
    print("OK batched rollout (lockstep == one-at-a-time; 3 episodes, 3 generate calls)")


def check_logprobs(model_name: str, device: str):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    with torch.no_grad():  # non-zero B so the adapter changes the output and every LoRA param gets a gradient
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    T = 600
    ids = torch.randint(0, model.config.get_text_config().vocab_size, (T,), device=device)
    mask = torch.zeros(T, dtype=torch.long, device=device)
    for a, b in [(120, 260), (300, 331), (400, 598)]:  # three generated spans, like assistant turns
        mask[a:b] = 1
    params = [p for p in model.parameters() if p.requires_grad]

    def run(fn):
        model.zero_grad(set_to_none=True)
        v = fn(model, ids, mask)
        v.backward()
        return v.item(), [p.grad.detach().clone() for p in params]

    v_ref, g_ref = run(reference_logprobs)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # the trainer's setting
    model.enable_input_require_grads()
    model.train()
    old_chunk = grpo.LOGPROB_CHUNK
    grpo.LOGPROB_CHUNK = 64  # 369 generated positions → 6 chunks, last one partial
    try:
        v_new, g_new = run(grpo.sequence_logprobs)
    finally:
        grpo.LOGPROB_CHUNK = old_chunk
    worst = max(((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item() for a, b in zip(g_new, g_ref))
    print(f"   value ref {v_ref:.6f} new (grad ckpt on) {v_new:.6f} · max rel grad diff {worst:.2e} over {len(params)} LoRA tensors")
    assert abs(v_new - v_ref) < 1e-4, "logprob value mismatch"
    assert worst < 1e-3, "LoRA gradient mismatch"
    print("OK sequence_logprobs")


def check_task_env(model_name: str):
    """run_train.task_env + evaluate on kind == "spider2" tasks (TPC-H SQLite file): the prompt names SQLite, run_sql hits
    the file, verify_spider judges, and a 2-seed eval reports per-seed accuracy (seed 1 submits the gold, seed 2 junk)."""
    import json as _json
    from lab import tasks
    from lab.tpch import SQLITE_CHECKS
    from train.run_train import evaluate, task_env
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tools = _json.loads(open("harness/tools.json").read())
    system_tpl = open("harness/system.md").read()
    tp = {x["id"]: x for x in _json.load(open("lab/tpch_tasks.json"))}
    ev = [tp[i] for i in SQLITE_CHECKS]
    s, tl, run_sql, vf = task_env(ev[0], system_tpl, tools, "")
    assert "SQLite" in s and "DuckDB" not in s and "DuckDB" not in _json.dumps(tl), "engine word not substituted"
    cols, rows, err = run_sql("select count(*) from lineitem")
    assert err is None and rows[0][0] == 600572, (cols, rows, err)
    assert vf(SQLITE_CHECKS[ev[0]["id"]])["pass"] and not vf("select 1")["pass"]
    s2, *_ = task_env(tasks.TASKS[0], system_tpl, tools, "")
    assert "DuckDB" in s2, "synth tasks must keep DuckDB"
    # --db-dir relocation must reach the verifier too: a task whose recorded db_path is bogus, relocated by schema name
    import copy, shutil, tempfile
    tmp = tempfile.mkdtemp(); shutil.copy(ev[0]["db_path"], f"{tmp}/{ev[0]['schema']}.sqlite")
    moved = copy.deepcopy(ev[0]); moved["db_path"] = "/nonexistent/laptop/path.sqlite"
    *_, vf2 = task_env(moved, system_tpl, tools, tmp)
    assert moved["db_path"].startswith(tmp) and vf2(SQLITE_CHECKS[ev[0]["id"]])["pass"], "db_dir relocation did not reach verify"
    try:
        task_env(dict(moved, db_path="/nonexistent/x.sqlite", schema="nosuchschema"), system_tpl, tools, tmp); raise AssertionError("missing db must raise")
    except FileNotFoundError:
        pass
    shutil.rmtree(tmp)
    call = lambda q: f"x</think>\n\n<tool_call>\n<function=submit>\n<parameter=sql>\n{q}\n</parameter>\n</function>\n</tool_call><|im_end|>"
    class P:  # seed is not visible in the prompt, so script by arrival order: the first len(ev) prompts are seed 1
        def __init__(self): self.n = 0
        def generate_batch(self, prompts, seeds):
            out = []
            for pr in prompts:
                tid = next(i for i in SQLITE_CHECKS if tp[i]["q"][:60] in pr)
                out.append(call(SQLITE_CHECKS[tid] if self.n < len(ev) else "select 1")); self.n += 1
            return out
    r = evaluate(P(), tok, system_tpl, tools, ev, max_turns=3, seeds=[1, 2], think=True)
    assert r["n"] == 4 and r["exec_acc"] == 0.5 and r["exec_acc_by_seed"] == {1: 1.0, 2: 0.0}, r
    print("OK task_env (spider2-kind tasks: SQLite prompt, file-backed run_sql, verify_spider; 2-seed eval 1.0 / 0.0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    check_reward()
    check_nudge()
    check_turn_rule(a.model)
    check_batched_rollout(a.model)
    check_task_env(a.model)
    check_logprobs(a.model, a.device)
    print("ALL OK")


if __name__ == "__main__":
    sys.exit(main())
