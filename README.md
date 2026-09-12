# sqlforge

> RL post-training of Qwen3.5-4B into a SQL query optimiser, using the FrogNano recipe.

**Tier: Tool.** Research training run; one operator, one artifact (a fine-tuned checkpoint + the environment that made it).

## Thesis

A 4B model post-trained purely by RL against a deterministic verifier, on tasks synthesised at its own
learnability frontier, can reach large-model quality on one narrow task. FrogNano showed this for
SWE-bench (4B → 61.5%). This project applies the same recipe to SQL query optimisation, where the
verifier is cleaner than tests: result-set identity plus measured cost.

## Recipe (from FrogNano, adapted)

- **Base model:** Qwen3.5-4B (the only model on which this exact recipe is demonstrated end to end).
  Optional second arm: granite-4.2-3B.
- **Verifier (gate):** rewritten query returns row-identical results to the original across N hidden data
  snapshots, including adversarial edge rows (NULLs, duplicates, empty tables). Failing the gate = reward 0.
- **Reward (shaped, success-gated):** cost reduction (EXPLAIN cost and/or measured latency) applied only to
  gate-passing rewrites. Failed exploration is never further penalised.
- **Task synthesis (TaskPilot-style):** generate schema + data + a deliberately slow query; roll out N
  trajectories from the *current* checkpoint; admit a task only if 0 < p̂ < 1, target 0.5; *rewrite* tasks
  outside the band rather than discarding them.
- **Harness:** minimal typed tools — `run_sql`, `explain`, `submit`. No bash. Natural termination: a turn
  with no tool call is the final answer.
- **RL:** group-relative (8 trajectories/task), tool observations masked from loss, no KL term, 0.5 partial
  credit for solved-but-truncated, success-gated log-length penalty.
- **Reward-hacking scaffold:** hidden snapshots the policy cannot read or modify; equivalence checked
  outside the sandbox.

## Ladder

1. Harness-fit smoke: base Qwen3.5-4B in the three-tool harness on 50 hand-made tasks. Measure turn-cap
   rate and gate pass rate before any training.
2. Environment + verifier: DuckDB/SQLite (start) → Postgres (later), snapshot generator, equivalence checker.
3. Task synthesis loop with the learnability gate.
4. GRPO/DPPO run, one 200-update climb; eval on a held-out task set.
5. Iterate climbs; watch for lost behaviours (FrogNano consolidation finding).

## Prior art (knowledge vault)

- `2026-09-09-frognano-4b-coding-agent-task-synthesis` — the recipe.
- `programmatic-rewards-for-grounded-structured-agents` — eval harness as reward; gate the degenerate answer.
- `2026-09-03-grpo-trl-ifstruct-schema-compliance` — cheapest runnable GRPO loop (T4, 100 steps).
- `2026-09-04-sql-tool-for-agents-lewis-ellis` — agents brute-force SQL in ways humans won't.
- `2026-07-04-harness-engineering-self-improvement` — evaluator outside the loop.
