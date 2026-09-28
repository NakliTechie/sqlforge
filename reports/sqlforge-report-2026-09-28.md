# sqlforge — final report (2026-09-24 → 2026-09-28)

**Tier: Tool.** One operator, one artifact: a GRPO training environment for multi-step analytical SQL plus the checkpoints it
produced. Status at close: the thesis is **not supported** by two training climbs and one ablation; one clean negative result, one measured
harness gap (10 points), one pre-registered analysis method, and a located cause (the pool, not the reward). GPU spend $101.73.

## 1. Objective and success criterion

Thesis (README): a 4B model post-trained purely by RL against a deterministic verifier, on tasks at its own learnability
frontier, reaches large-model quality on multi-step analytical SQL. Pre-registered success criterion (soc 2026-09-27 10:35,
amended 11:10 after cold review): **beat the base Qwen3.5-4B on the Spider 2.0-Lite SQLite slice (135 tasks) under our
harness, paired per task, with a 30-database cluster-bootstrap 95 % CI excluding zero and Δ ≥ +5 points.** Secondary: BIRD
Mini-Dev (496 tasks, in-family). Analysis code: `lab/judge_stats.py` (paired Δ, cluster bootstrap primary; exact McNemar and
database sign-flip secondary; seen/unseen-schema and base-reachable/never strata; transition decomposition; SQL shape).

## 2. What was built

- **Environment.** Two tools (`run_sql`, `submit`), SQLite databases, 25-turn cap, thinking on, 2,048 tokens per turn.
  Verifier `lab/spider2.py`: Spider column-vector rules (condition columns, ignore-order, 1e-2 tolerance, NULL→0) and BIRD
  multiset match; read-only connections with a 60-second statement timeout.
- **Trainer** (`train/`): GRPO, LoRA r32 on Qwen3.5-4B bf16, in-process vLLM rollouts with LoRA remap, 8 tasks × 8 rollouts
  per step, group-relative advantages, zero-variance groups dropped, outcome reward pass 1 / wrong 0 / no-submit −1,
  success-gated log-length penalty (inert in practice: 83 % of passes were under the 6,000-character target), dynamic
  sampling (rest a task after 2 zero-variance groups, re-admit after 25 steps), OOM-skip in the backward, 32k context guard,
  checkpoint-before-eval, resumable checkpoints in a bucket.
- **Pools.** Climb 1: 70 synthesised hop-3/4 tasks. Climb 2: 521 tasks with measured base pass@8 in (0, 1): 469 BIRD-train
  (difficulty proxy ≥ 4, `lab/bird_train.py`), 25 TPC-DS (95 authored questions, `lab/tpcds_questions.py`), 27 TPC-H, over
  57 SQLite databases.
- **Ops** (`infra/`, `~/Code/infra`): spot RTX PRO 6000 (g4-standard-48, $1.77/h), relaunch loops under launchd, spend caps,
  self-deleting VMs, DONE markers, two-tier supervision, a per-life cost ledger. Lessons went to the public
  `NakliTechie/remote-gpu-guide` (chapter 7, §4.8b–4.10).

## 3. Results

### 3.1 Climb 1 (Experiment 7–8): the synthetic pool saturated and transferred nothing
100 steps on 70 synthetic tasks, $23.60. Train pass 0.88 → 0.98; held-out synthetic eval 0.88 → 0.98–1.00; groups with
gradient 3.2 → 0.9 per step. Spider judge: step100 × 3 seeds vs base × 3, Δ +1.73, CI [−1.57, +4.98] — null.
Lesson: the pool, not the recipe, was the limit. Base pass@8 measurements (Experiments 9–11, $6.36) found the learnable band:
Spider 50/135 learnable, 80 never in 8 samples; TPC-DS is Spider's statistical twin (pass@1 0.170 vs 0.165, 61 % vs 59 %
never); BIRD-train ≈ 40 % learnable at proxy ≥ 4.

### 3.2 Climb 2 (Experiment 12): in-family gain, on-target loss
150 steps on the 521-task pool, 11 VM lives, $33.44. Steering eval (BIRD-challenging × 3 seeds, in-process): 0.406 → 0.535
(step 100) → 0.502 (step 150); no-submit 0.063 → 0.013, turns 10.9 → 7.7.

Judge, lab.run harness (server vLLM, one tool message per call, prior thinking dropped), base × 8 seeds = 0.169:

| checkpoint | seeds | acc | Δ | 30-DB CI | McNemar p |
|---|---|---|---|---|---|
| step60 | 5 | 0.154 | −1.54 | [−4.16, +0.95] | 0.80 |
| step80 | 5 | 0.181 | +1.13 | [−0.86, +2.74] | 0.51 |
| step100 | 5 | 0.184 | +1.43 | [−1.12, +4.28] | 1.00 |
| **step150 (primary)** | 5 | **0.141** | **−2.87** | **[−6.16, +0.35]** | 0.12 |

**Success criterion not met.** BIRD Mini-Dev secondary (in-family): step150 × 3 vs base × 3 = 0.514 → 0.595, Δ +8.10,
CI [+4.92, +11.64], McNemar p < 0.001; base-never stratum 0 → 18.1.

### 3.3 The harness gap (Experiment 12, in-process diagnostic): 10 points on the base
Cold review (Opus, C1) asked whether the judge harness matched the training harness. Measured on the same base weights,
same 135 tasks, same verifier, 3 seeds: **0.269 under the trainer's rollout path** (prior thinking kept in context,
top_p 0.95, T 0.6, merged tool messages) vs **0.169 under lab.run**. Paired Δ +9.97, CI [+6.92, +13.81], McNemar p 0.001,
seen and unseen schemas alike. Every earlier Spider verdict was measured in a harness that caps the base 10 points below
its in-harness ability.

### 3.4 Climb 2 in its own harness: worse than the base
step150 × 3 seeds under the trainer's path: **0.200 vs base 0.269**, Δ **−6.91**, CI [−13.06, −1.56], McNemar p 0.001,
DB sign-flip p 0.021. Base-reachable stratum 61.6 → 35.0 (−26.6); base-never 0 → 8.3. No-submit 0.375 → 0.193, turns
19.8 → 15.5. With the harness confound removed, climb 2 made the policy worse on the target.

### 3.5 Reading
The BIRD gain and the Spider loss are one behaviour change: the policy learned to submit earlier and more often. That pays on
BIRD's short questions (no-submit → wrong→pass transitions 12.6 % of task mass) and destroys the patient exploration the base
uses on analytical schemas (pass→wrong 5.0 % on BIRD; base-reachable −26.6 on Spider). Two candidate causes, pre-registered
for the ablation: the commitment reward (no-submit −1, flagged by all four cold reviews) and the pool composition (90 % BIRD).

## 4. Cold reviews (2026-09-27)
codex, DeepSeek reasoner, Claude Opus 5.5 subagent, opencode/space-bunny; brief and responses in `reports/review-*.md`,
`reports/review-response-2026-09-27.md`. Accepted and acted on: pre-registered judge analysis with a database-clustered test,
per-task logging, harness-parity diagnostic (which produced §3.3–3.4), seen-schema stratum, OOM-skip. Accepted, not yet built:
temperature-consistent log-probs, length normalisation, strict training verifier, named large-model comparator, 2 × 75-step
design. The reviews' central forecast — in-family gain that does not transfer — held.

## 5. Costs
GPU ledger (`~/Code/infra/gcp/gpu-usage.csv`): 38 sqlforge lives, 52.40 GPU-hours, **$101.73** (climb 1
$23.60 + judge $7.07; pass@8 $6.36; climb 2 $33.44 + judge $7.12 + judge life 2 $9.09; ablation 1 $8.98), 39 lives, 57.47 GPU-hours. Incidents: 9
across both climbs (bucket paths, verifier path relocation, 32k context, duplicate VM on a failed listing, OOM at step 146,
per-boot campaign paths, needless relaunch, serial-SQL stall, `..` in gs:// paths); each has a fix in the code and a line in the
runbook. Cold reviews ≈ $0.15 (DeepSeek) + subscriptions.

## 6. Experiment 13 — Ablation 1: the commitment reward
Pre-registered 2026-09-28 (leg Experiment 13) before launch. Climb-2 recipe unchanged except `--no-submit-reward 0`
(a group of only {wrong, no-submit} is then zero-variance and dropped), 40 steps, no steering evals; judged in the trainer's
harness (adapter arm × 3 seeds vs the existing base arm 0.269). Decision rule: Δ ≥ 0 and base-reachable ≥ −5 clears the reward
and indicts the pool; base-reachable ≤ −15 reproduces the collapse without the penalty and indicts the pool; in between is
inconclusive. Launched 18:54 IST as `sqlforge-ablate1`, cap $12.

**Result (2026-09-29 00:05 IST, 1 life, 304 min, $8.98).** Training side indistinguishable from climb 2's first 40 steps (train pass
0.660 vs 0.658, no-submit 0.049 vs 0.040). Judge, trainer's harness, step40 × 3 vs base × 3: **0.205 vs 0.269**, Δ **−6.42**,
CI [−11.03, −2.58], McNemar p 0.004; **base-reachable 61.6 → 42.9 (−18.6)**; base-never 0 → 3.1; no-submit 0.506 (base 0.375,
climb-2 step150 0.193), turns 21.7. Against climb-2 step150 in the same harness: Δ +0.49 [−2.99, +4.46] — the same loss.
**Decision rule (ii): the collapse reproduces without the commitment penalty. The pool, 90 % BIRD-train, is the cause; the reward
is cleared.** The two runs fail by opposite routes — early wrong submissions with the penalty, explore-to-the-cap abstention
without it — and lose the same tasks: the ones the base could already solve. Forty steps on BIRD-majority data are enough to
displace the base's analytical-schema competence; 150 steps do not make it worse.

## 7. What would change next (not launched; project closed after the ablation)
1. Judge in the trainer's harness, pre-registered, with a named large-model comparator in the same harness.
2. Reward: drop the commitment penalty; per-trajectory length-normalised, temperature-consistent log-probs; strict verifier.
3. Pool: the ablation makes this the first-order change — TPC-DS/TPC-H majority with regenerated parameter variants, BIRD at most a
   minority, and a held-out analytical dev set for checkpoint selection, never the steering set.
4. Trainer throughput: threaded tool calls (done, `72d3af8`), incremental prompt-length tracking, parallel verification.
5. Design: 2 × 75 steps with two seeds before any 150-step run.

## 8. Artifacts
- Code: `NakliTechie/sqlforge` main (`train/`, `lab/`, `infra/`, `harness/`).
- Results mirrors: `runs/spider2-eval/` (climb 1 judge), `runs/spider2-eval2/`, `runs/spider2-eval2b/` (climb 2 judge and
  diagnostic), `runs/passk{,2,3}/`, `runs/climb2/`, `runs/ablate1/`.
- Adapters: `gs://sqlforge-bf3e24-smoke/climb2/saves/step{20..150}`, `ablate1/ckpt/step40` — mirrored locally before billing
  is disconnected (section 9).
- Plan and lab notebook: `plan/lab/task-synth/2026-09-24-leg.md` (Experiments 1–13), `plan/soc.md`, `plan/pending.md`.

## 9. Close-out
Results mirrored under `runs/`. Adapters stay in `gs://sqlforge-bf3e24-smoke` (trimmed of optimizer states and re-downloadable
databases, ≈ 4 GB ≈ $0.08/month) with billing linked; in December they move bucket-to-bucket to the new GCP account (procedure:
`~/.claude/delegations/sqlforge/closeout.sh howto`). Chirag, 2026-09-28 21:05: nothing pulled to the laptop; revisit billing in December.
