"""Group-relative policy optimisation, FrogNano-shaped:
- G trajectories per task, rewards standardised within the group, advantage broadcast to generated tokens
- zero-variance groups dropped
- tool observations in context but masked from the loss
- no reference-model KL, no entropy bonus; gradient clipping for stability
- success-gated log-length penalty (only on successful, completed trajectories)
- per-trajectory token mean, then batch mean ("prompt_mean" — long rollouts must not own the gradient)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def shaped_reward(passed: bool, hit_cap: bool, gen_chars: int, target_chars: int, alpha: float) -> float:
    if not passed:
        return 0.0
    if hit_cap:
        return 0.5
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


def sequence_logprobs(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean log-prob of generated tokens for one sequence (ids: [T], mask: [T])."""
    inp = ids[:-1].unsqueeze(0)
    tgt = ids[1:]
    m = mask[1:].float()
    logits = model(input_ids=inp).logits[0].float()
    lp = torch.gather(F.log_softmax(logits, dim=-1), 1, tgt.unsqueeze(1)).squeeze(1)
    return (lp * m).sum() / m.sum().clamp(min=1.0)


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
