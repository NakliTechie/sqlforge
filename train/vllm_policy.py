"""vLLM rollout policy: batched, prefix-cached generation on the same GPU as the HF trainer.

Same contract as HFPolicy.generate_batch(prompts, seeds) -> texts (raw continuations of raw prompt text, special
tokens kept so <tool_call> XML and </think> survive). Weights follow the trainer through LoRA: after each GRPO step
the trainer saves its PEFT adapter and calls set_adapter(path, step); vLLM loads it as a new LoRA version, so the base
weights are never copied. Before the first update (step 0) rollouts come from the base model.

`logprob_gap` measures trainer/engine agreement on the same tokens (the vault's Mercor note treats < 0.03 as healthy).
"""
from __future__ import annotations


class VLLMPolicy:
    def __init__(self, model: str, *, max_new_tokens: int = 2048, temperature: float = 0.6, top_p: float = 0.95,
                 gpu_memory_utilization: float = 0.35, max_model_len: int = 32768, max_lora_rank: int = 16,
                 enforce_eager: bool = False):
        from vllm import LLM
        self.llm = LLM(model=model, dtype="bfloat16", enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1,
                       enable_prefix_caching=True, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, enforce_eager=enforce_eager)
        self.max_new_tokens, self.temperature, self.top_p = max_new_tokens, temperature, top_p
        self.lora = None

    def set_adapter(self, path: str, version: int) -> None:
        from vllm.lora.request import LoRARequest
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
