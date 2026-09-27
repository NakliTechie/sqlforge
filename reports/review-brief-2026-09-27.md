# sqlforge — cold review brief (2026-09-27 10:00 IST)

You are an independent reviewer. Critique the METHOD and the PROGRESS CLAIMS below as a sceptical ML researcher who has
seen many RL-for-LLM projects fool themselves. Be specific and severe. We want: (1) flaws or confounds that would make the
step-50 result not mean what we think; (2) statistical weaknesses; (3) reward/verifier/pool design errors; (4) what the
Spider result at the end can and cannot establish; (5) the three highest-value changes for the next climb. Do not praise.
Every claim you make should point at a file/line or a number in this brief.

## Thesis and success criterion
A 4B model (Qwen3.5-4B) post-trained purely by RL (GRPO, LoRA r32) on synthesized/curated frontier tasks reaches
large-model quality on multi-step analytical SQL. Pre-registered success criterion: beat the base model on the Spider 2.0-
Lite SQLite slice (135 tasks, 30 real databases, official result-match rules), measured with the same agentic harness.

## The system
- Harness: two tools (`run_sql` → up to 20 rows / 1,500 chars of observation; `submit` → final SQL), 25-turn cap, Qwen
  thinking on, 2,048 tokens/turn, one turn rule shared by measurement and training (harness/parse.py). Schema DDL in the
  system prompt; a reference document appended to the question when the benchmark provides one.
- Reward (train/grpo.py): pass=1, wrong=0, no-submit=−1 (SkyRL-SQL style), success-gated log-length penalty (alpha 0.1).
  Group-relative advantages over 8 rollouts per task; groups with one outcome class are dropped (no gradient).
- Rollouts: in-process vLLM with the LoRA adapter re-exported every step (train/vllm_policy.py); tool observations masked
  out of the loss; 8 tasks × 8 rollouts per step; lr 1e-5; gradient checkpointing.
- Verifier for real-schema tasks (lab/spider2.py): execute on the one SQLite DB, compare to gold rows with Spider 2.0's
  rules (gold columns must each appear as some predicted column vector, row counts equal, numeric tol 1e-2, NULL→0,
  order only when the gold has ORDER BY). BIRD tasks use BIRD's exact-row-set EX. Golds for TPC-H/TPC-DS come from DuckDB
  on the same generated data exported to SQLite.

## Climb 1 (2026-09-25/26) — negative result
Pool: 70 synthesized DuckDB tasks (hops 3–4) on toy schemas with hidden data snapshots. 100 GRPO steps, $23.60.
In-distribution eval 0.88 → 0.98–1.00 (saturated by step 75); groups with gradient per step 3.24 → 0.92 by quarter.
Outside evals (3 seeds each, base vs step100): Spider 2.0 slice 0.170 (26/22/21 of 135) vs 0.188 (24/27/25); BIRD Mini-Dev
0.514 (262/248/258 of 498) vs 0.517 (250/254/269). Every checkpoint inside the base's seed range. Behaviour did change:
no-submit on Spider 0.33 → 0.18 at equal turns. Diagnosis: pool difficulty far below target; no outcome variance where it
mattered.

## Pool measurement (2026-09-26/27)
Base pass@8 (8 samples, T=0.6) to locate the learnable band (0 < p̂ < 1):
| set | tasks | pass@1 | pass@8 | learnable | never | always |
| Spider 2.0 slice | 135 | 0.165 | 0.407 | 50 | 80 | 5 |
| TPC-DS sf0.1 (72 of 99 queries, 95 hand-authored NL questions) | 72 | 0.170 | 0.389 | 25 | 44 | 3 |
| TPC-H sf0.1 (22 queries + parameter variants) | 47 | 0.519 | 0.809 | 27 | 9 | 11 |
| BIRD-train candidates, batch 1 (gold-SQL difficulty proxy ≥ 4) | 577 | 0.376 | 0.558 | 225 | 255 | 97 |
| BIRD-train candidates, batch 2 (same proxy, disjoint) | 577 | ~0.47 | — | 244 | — | — |
TPC-DS matches the Spider slice's profile almost exactly (pass@1, pass@8, never share, ~20 turns, ~500 s/episode).
BIRD has a documented ~50 % annotation-error rate; "never" tasks likely include wrong golds. The proxy did NOT predict
learnability inside the ≥ 4 slice (28–50 % learnable at every score).

## Climb 2 (running since 2026-09-27 01:12 IST)
Pool (lab/climb2_pool.json): 521 tasks with 0 < base p̂ < 1: 469 BIRD-train (train split, DBs disjoint from Mini-Dev),
25 TPC-DS, 27 TPC-H; 57 databases; p̂ quartiles 0.25 / 0.50 / 0.75. Dynamic sampling: a task whose last 2 groups had no
outcome variance is rested for 25 steps. 150 steps planned. Steering eval every 25 steps: BIRD Mini-Dev "challenging"
(101 tasks) × 3 seeds. The Spider 135 is untouched until the end (final judge, base + every 20th-step adapter, step-150 × 3).
Training signal: groups with gradient per step 5.3 (steps 1–25), 5.0 (26–49); train pass 0.65–0.68; ~5.5 min/step.
Steering eval so far (mean of 3 seeds; per-seed):
  step 0   0.406  (0.337 / 0.436 / 0.446)  no-submit 0.063  turns 10.9
  step 25  0.472  (0.465 / 0.505 / 0.446)  no-submit 0.056  turns 11.2
  step 50  0.495  (0.505 / 0.505 / 0.475)  no-submit 0.066  turns 13.5   (eval wall 836 s → 1,720 s)
We called step 50 "movement confirmed" because every seed is above the baseline's best seed and the curve is monotone.
Incidents: 3 boot/eval crashes (bucket path, verifier path, a 32,769-token prompt → context guard added), ≈ $2.5.

## Things we already worry about (tell us which are real and what we missed)
1. Same-family eval: BIRD-train (pool) and BIRD Mini-Dev challenging (steering eval) share annotators, question style and
   the "evidence" hint convention, though not databases. Is +9 points here mostly style adaptation? Spider is the judge, but
   we also select checkpoints/stop rules by this eval.
2. 3 seeds on 101 tasks: the step-0 seed spread was 11 points. Is "all seeds above the baseline's best seed" a sound rule?
3. Turns rose 10.9 → 13.5 and eval wall time doubled. Is the policy learning to explore, or learning to stall toward the
   cap where no-submit = −1 dominates? The success-gated length penalty only applies to passes.
4. Label noise in BIRD golds: RL against ~50 % wrong golds — the band filter removes never-pass tasks, but a wrong gold that
   the model sometimes matches by accident stays in the pool. How much does this cap or distort learning?
5. Verifier leniency: Spider's column-vector matching accepts extra columns; NULL→0; numeric tolerance 1e-2 absolute.
   Reward hacking surfaces?
6. GRPO specifics: no KL/reference term, no clipping ratio mentioned here (check train/grpo.py), lr 1e-5 constant, LoRA
   r32; 64 rollouts/step; group filter drops ~35 % of groups. Is this a sound estimator at this scale?
7. Dynamic sampling rests tasks after 2 zero-variance groups — could this systematically remove the hardest (0/8) tasks and
   leave an easier pool, inflating in-run pass without transfer?
8. The pass@8 band was measured on the BASE policy; the band moves during training (we re-admit after 25 steps). Is that
   enough, or should we re-measure p̂ under the current policy?
9. What would a convincing final Spider result look like (base 0.165, pass@8 ceiling 0.41, 3-seed noise ±5 tasks)? What
   effect size and design (seeds, paired tests) would you demand before calling the thesis supported?

## Pointers (repo: ~/Code/sqlforge; plan/ is a symlink into ~/Code/plans/sqlforge)
- Lab record with every experiment and table: plan/lab/task-synth/2026-09-24-leg.md (Experiments 7–11 + climb-2 log at end)
- Incident/decision log: plan/soc.md (newest first)
- Trainer: train/run_train.py (task_env, evaluate, dynamic sampling, step loop), train/grpo.py (reward, advantages,
  logprobs), train/rollout.py (Episode, run_episodes, context guard), train/vllm_policy.py
- Harness: harness/loop.py, harness/parse.py, harness/system.md, harness/tools.json, harness/nudge.py
- Verifiers: lab/verify.py (compare, G0–G4), lab/spider2.py (verify_spider, build_bird)
- Pools: lab/tpch.py, lab/tpcds.py + lab/tpcds_questions.py, lab/bird_train.py, lab/pool_from_passk.py, lab/climb2_pool.json
- Results mirrors: runs/spider2-eval/ (climb-1 outside eval), runs/passk/, runs/passk2/, runs/passk3/ (pass@8), runs/climb1/
- Prior-art report: reports/RL for agentic text to SQL.md
