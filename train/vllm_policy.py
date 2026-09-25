"""vLLM rollout policy: batched, prefix-cached generation on the same GPU as the HF trainer.

Same contract as HFPolicy.generate_batch(prompts, seeds) -> texts (raw continuations of raw prompt text, special
tokens kept so <tool_call> XML and </think> survive). Weights follow the trainer through LoRA: after each GRPO step
the trainer saves its PEFT adapter and calls set_adapter(path, step); vLLM loads it as a new LoRA version, so the base
weights are never copied. Before the first update (step 0) rollouts come from the base model.

Qwen3.5 checkpoints are vision-language models. HF's AutoModelForCausalLM builds the text-only Qwen3_5ForCausalLM
(LoRA keys `base_model.model.model.layers.*`); vLLM defaults to Qwen3_5ForConditionalGeneration, whose language
layers live under `model.language_model.*`, so an unmodified HF adapter matches nothing there and is silently a
no-op (GCP run gpu4, 2026-09-25: vLLM vs HF logprob gap 0.60 = the adapter's whole effect). Two fixes, `lora_mode`:
- "remap": keep vLLM's default class; rewrite the adapter keys to the VL layout before loading.
- "textonly": force vLLM's text-only Qwen3_5ForCausalLM (hf_overrides), whose mapper matches HF's keys as they are.
train/check_vllm.py measures which one makes vLLM agree with the trainer (< 0.03 mean |Δ logprob|).
"""
from __future__ import annotations

import os
import shutil

HF_PREFIX = "base_model.model.model.layers."
VL_PREFIX = "base_model.model.model.language_model.layers."


def export_adapter(peft_dir: str, out_dir: str) -> str:
    """Copy a PEFT adapter with its layer keys moved to the VL layout (for vLLM's ConditionalGeneration class)."""
    from safetensors.torch import load_file, save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = load_file(os.path.join(peft_dir, "adapter_model.safetensors"))
    save_file({k.replace(HF_PREFIX, VL_PREFIX, 1): v for k, v in sd.items()},
              os.path.join(out_dir, "adapter_model.safetensors"))
    shutil.copy(os.path.join(peft_dir, "adapter_config.json"), out_dir)
    return out_dir


class VLLMPolicy:
    def __init__(self, model: str, *, max_new_tokens: int = 2048, temperature: float = 0.6, top_p: float = 0.95,
                 gpu_memory_utilization: float = 0.35, max_model_len: int = 32768, max_lora_rank: int = 16,
                 enforce_eager: bool = False, lora_mode: str = "remap"):
        from vllm import LLM
        assert lora_mode in ("remap", "textonly"), lora_mode
        extra = {"hf_overrides": {"architectures": ["Qwen3_5ForCausalLM"]}} if lora_mode == "textonly" else {}
        self.llm = LLM(model=model, dtype="bfloat16", enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1,
                       enable_prefix_caching=True, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, enforce_eager=enforce_eager, **extra)
        self.max_new_tokens, self.temperature, self.top_p, self.lora_mode = max_new_tokens, temperature, top_p, lora_mode
        self.lora = None

    def set_adapter(self, path: str, version: int) -> None:
        from vllm.lora.request import LoRARequest
        if self.lora_mode == "remap":
            path = export_adapter(path, path.rstrip("/") + "-vllm")
        self.lora = LoRARequest(f"step{version}", version, path)

    def _params(self, seed, **kw):
        from vllm import SamplingParams
        return SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_new_tokens,
                              seed=seed, skip_special_tokens=False, **kw)

    def generate_batch(self, prompts: list[str], seeds: list[int | None]) -> list[str]:
        outs = self.llm.generate(prompts, [self._params(s) for s in seeds], lora_request=self.lora, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def generate(self, prompt_text: str) -> str:
        return self.generate_batch([prompt_text], [None])[0]

    def sample_with_logprobs(self, prompt: str, seed: int = 0) -> tuple[str, list[int], list[float]]:
        o = self.llm.generate([prompt], [self._params(seed, logprobs=0)], lora_request=self.lora, use_tqdm=False)[0].outputs[0]
        return o.text, list(o.token_ids), [lp[t].logprob for t, lp in zip(o.token_ids, o.logprobs)]

