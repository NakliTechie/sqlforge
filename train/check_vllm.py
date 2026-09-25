"""GPU check for the vLLM rollout policy (run on a CUDA box):

1. vLLM loads the base model with LoRA enabled and generates.
2. A PEFT adapter saved by the HF trainer (random non-zero B, so it changes the model) loads into vLLM via
   set_adapter, and the adapter measurably changes vLLM's logprobs of the same tokens.
3. Trainer/engine agreement: on vLLM's sampled tokens, per-token logprobs from vLLM and from the HF+PEFT model
   agree (mean |Δ| < 0.03, the vault Mercor note's healthy threshold).

uv run python -m train.check_vllm --model Qwen/Qwen3.5-0.8B
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile

import torch

from lab import tasks
from train.vllm_policy import VLLMPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--lora-r", type=int, default=16)
    a = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    t = tasks.TASKS[0]
    tools = json.load(open("harness/tools.json"))
    prompt = tok.apply_chat_template([{"role": "system", "content": "You write DuckDB SQL.\n" + t["ddl"]},
                                      {"role": "user", "content": t["q"]}], tools=tools, tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda")
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    model.eval()
    d = tempfile.mkdtemp()
    model.save_pretrained(d)
    # engine after the HF load (vLLM's config registration breaks a later HF from_pretrained of Qwen3.5)
    policy = VLLMPolicy(a.model, max_new_tokens=128, gpu_memory_utilization=0.35, max_lora_rank=max(16, a.lora_r))
    base_text, _, _ = policy.sample_with_logprobs(prompt, seed=1)
    print("CHECK base generate ok:", repr(base_text[:80]), flush=True)
    policy.set_adapter(d, 1)
    text, gen_ids, v_lp = policy.sample_with_logprobs(prompt, seed=2)
    print("CHECK adapter generate ok:", repr(text[:80]), f"({len(gen_ids)} tokens)", flush=True)

    p_ids = tok(prompt, add_special_tokens=False).input_ids
    ids = torch.tensor([p_ids + gen_ids], device="cuda")
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, len(p_ids) - 1:-1].float()
        hf_lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(gen_ids, device="cuda")[:, None]).squeeze(1).cpu()
        with model.disable_adapter():
            base_logits = model(input_ids=ids).logits[0, len(p_ids) - 1:-1].float()
        base_lp = torch.log_softmax(base_logits, -1).gather(1, torch.tensor(gen_ids, device="cuda")[:, None]).squeeze(1).cpu()
    v = torch.tensor(v_lp)
    gap = (v - hf_lp).abs().mean().item()
    adapter_effect = (hf_lp - base_lp).abs().mean().item()
    res = {"tokens": len(gen_ids), "mean_abs_gap_vllm_vs_hf": round(gap, 4), "adapter_effect_hf": round(adapter_effect, 4),
           "vllm_mean_lp": round(v.mean().item(), 4), "hf_mean_lp": round(hf_lp.mean().item(), 4)}
    ok = gap < 0.03 and adapter_effect > 5 * gap
    print("CHECK", json.dumps(res), "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
