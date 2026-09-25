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
    tools = _json.loads(open("harness/tools.json").read())
    tr = hf_episode(P(), tok, "S", t["q"], tools, run_sql, max_turns=10, think=True)
    assert tr.submitted_sql == gold and tr.turns == 3 and not tr.hit_cap, (tr.submitted_sql, tr.turns)
    gen = [txt for txt, g in tr.segments if g]
    assert len(gen) == 3 and parse.EMPTY_TURN_NOTE in tr.segments[2][0]
    ids, mask = tokenize_trajectory(tok, tr)
    assert tok.decode(ids[mask.bool()]) == "".join(gen)
    print("OK turn rule (loop.py ollama + openai and rollout.py agree: note, thinking-call, submit)")


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
    old_chunk = grpo.LOGPROB_CHUNK
    grpo.LOGPROB_CHUNK = 64  # 369 generated positions → 6 chunks, last one partial
    try:
        v_new, g_new = run(grpo.sequence_logprobs)
    finally:
        grpo.LOGPROB_CHUNK = old_chunk
    worst = max(((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item() for a, b in zip(g_new, g_ref))
    print(f"   value ref {v_ref:.6f} new {v_new:.6f} · max rel grad diff {worst:.2e} over {len(params)} LoRA tensors")
    assert abs(v_new - v_ref) < 1e-4, "logprob value mismatch"
    assert worst < 1e-3, "LoRA gradient mismatch"
    print("OK sequence_logprobs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    check_reward()
    check_nudge()
    check_turn_rule(a.model)
    check_logprobs(a.model, a.device)
    print("ALL OK")


if __name__ == "__main__":
    sys.exit(main())
