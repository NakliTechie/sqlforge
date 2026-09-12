# sqlforge

> RL post-training of Qwen3.5-4B into a multi-step analytical SQL agent (question → decompose → query → assemble), using the FrogNano recipe.

**Tier: Tool.** Research training run; one operator, one artifact (a fine-tuned checkpoint + the environment that made it).

## Thesis

A 4B model post-trained purely by RL against a deterministic verifier, on tasks synthesised at its own
learnability frontier, can reach large-model quality on one narrow task. FrogNano showed this for
SWE-bench (4B → 61.5%). This project applies the recipe to **multi-step analytical SQL**: a question over a
schema that the agent must decompose, explore with queries, and answer with one final SQL. Writing a single
SQL statement is not the hard part — planning the steps and assembling them is.

## Recipe (from FrogNano, adapted)

- **Base model:** Qwen3.5-4B (the only model on which this exact recipe is demonstrated end to end).
- **Harness:** two typed tools — `run_sql` (scratch queries, truncated observations) and `submit` (the final
  SQL). Natural termination: a turn with no tool call is the final answer.
- **Verifier — a six-gate deterministic ladder**, one module, three callers (task admission, rollout reward,
  held-out eval): G0 `sqlglot` parse → G1 `sqlglot` qualify against schema → G2 `sqlfluff` lint (diagnostics
  only) → G3 DuckDB execute with timeout/row cap → G4 result match vs gold on **N hidden snapshots** (Spider
  2.0 rules: `condition_cols`, `ignore_order`, `abs_tol=1e-2`, multiple golds) → G5 cost (parked).
  Reward is binary on G0–G4. The submission must be SQL, run on hidden data — a hardcoded answer fails.
- **Task synthesis (TaskPilot-style):** generate schema + snapshot generator + question + gold SQL with a
  **hop count** (1–5 dependent steps) as the difficulty knob; admit a task only if its gold passes the ladder
  and the current checkpoint solves it with 0 < p̂ < 1 (target 0.5); rewrite tasks outside the band.
- **RL:** group-relative (8 trajectories/task), tool observations masked from loss, no KL term, 0.5 partial
  credit for solved-but-truncated, success-gated log-length penalty.

## Ladder

1. Harness-fit smoke (lab campaign 1): base Qwen3.5-4B in the harness on 30 multi-hop tasks. exec_acc,
   per-gate failures, turn-cap rate before any training.
2. Environment hardening; Spider 2.0 SQLite/DuckDB slices wired as outside held-out eval.
3. Task synthesis loop with the learnability gate.
4. GRPO/DPPO run, one 200-update climb; eval on held-out + Spider 2.0 slices.
5. Iterate climbs; watch for lost behaviours (FrogNano consolidation finding).

## Prior art (knowledge vault)

- `2026-09-09-frognano-4b-coding-agent-task-synthesis` — the recipe.
- `programmatic-rewards-for-grounded-structured-agents` — eval harness as reward; gate the degenerate answer.
- `2026-09-03-grpo-trl-ifstruct-schema-compliance` — cheapest runnable GRPO loop (T4, 100 steps).
- `2026-09-04-sql-tool-for-agents-lewis-ellis` — agents brute-force SQL in ways humans won't.
- `2024-11-12-spider-2-enterprise-text-to-sql-workflows` — the outside benchmark and the result-match rules.
- `2026-07-04-harness-engineering-self-improvement` — evaluator outside the loop.
