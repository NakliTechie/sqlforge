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
    check_logprobs(a.model, a.device)
    print("ALL OK")


if __name__ == "__main__":
    sys.exit(main())
