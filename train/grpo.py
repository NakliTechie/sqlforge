"""Group-relative policy optimisation, FrogNano-shaped:
- G trajectories per task, rewards standardised within the group, advantage broadcast to generated tokens
- zero-variance groups dropped
- tool observations in context but masked from the loss
- no reference-model KL, no entropy bonus; gradient clipping for stability
- success-gated log-length penalty (only on successful trajectories)
- per-trajectory token mean, then batch mean ("prompt_mean" — long rollouts must not own the gradient)
- memory: logits only at positions that predict a generated token, log-softmax in checkpointed row chunks
"""
from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint

LOGPROB_CHUNK = 1024  # rows of [chunk, vocab] fp32 alive at once during log-softmax (fwd and recompute)


def shaped_reward(passed: bool, gen_chars: int, target_chars: int, alpha: float) -> float:
    """Binary pass, discounted by a log-length penalty past target_chars. A capped episode never submitted, so it
    cannot pass; there is no partial credit."""
    if not passed:
        return 0.0
    if gen_chars <= target_chars or alpha <= 0:
        return 1.0
    return max(0.0, 1.0 - alpha * math.log(gen_chars / target_chars))


def group_advantages(rewards: list[float]) -> list[float] | None:
    mu = sum(rewards) / len(rewards)
    var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    if var < 1e-8:
        return None
    sd = math.sqrt(var)
    return [(r - mu) / sd for r in rewards]


def _chunk_logprob_sum(logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    lf = logits.float()
    return (lf.gather(1, tgt.unsqueeze(1)).squeeze(1) - torch.logsumexp(lf, dim=-1)).sum()


def sequence_logprobs(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean log-prob of generated tokens for one sequence (ids: [T], mask: [T]).

    Only the K positions that predict a generated token get logits (`logits_to_keep`), so the [T, vocab] tensor
    never exists; the fp32 log-softmax runs over checkpointed row chunks, so autograd keeps the bf16 [K, vocab]
    logits plus one fp32 chunk instead of fp32 [T, vocab] twice."""
    keep = mask[1:].nonzero(as_tuple=True)[0]
    if keep.numel() == 0:
        raise ValueError("trajectory has no generated tokens")
    logits = model(input_ids=ids[:-1].unsqueeze(0), logits_to_keep=keep).logits[0]  # [K, V]
    tgt = ids[1:][keep]
    total = logits.new_zeros((), dtype=torch.float32)
    for s in range(0, keep.numel(), LOGPROB_CHUNK):
        total = total + checkpoint(_chunk_logprob_sum, logits[s:s + LOGPROB_CHUNK], tgt[s:s + LOGPROB_CHUNK],
                                   use_reentrant=False)
    return total / keep.numel()


def grpo_step(model, optimizer, batch: list[tuple[torch.Tensor, torch.Tensor, float]], device, grad_clip=1.0) -> dict:
    """batch: list of (ids, mask, advantage) for trajectories from non-degenerate groups. On-policy, one
    update per batch, so the importance ratio is 1 and the clipped objective reduces to -A * logp."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for ids, mask, adv in batch:
        lp = sequence_logprobs(model, ids.to(device), mask.to(device))
        loss = -(adv * lp) / len(batch)
        loss.backward()
        total += loss.item()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return {"loss": total, "grad_norm": float(gn)}
