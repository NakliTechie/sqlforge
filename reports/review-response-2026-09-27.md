# Response to the cold reviews (2026-09-27, 11:15 IST)

Four independent reviews of `reports/review-brief-2026-09-27.md`: codex (agentic, read-only), DeepSeek reasoner (bundle),
Claude Opus 5.5 (agentic subagent, read after the first two), opencode/space-bunny (agentic, recomputed statistics).
Grok (expired login) and kimchi's hosted Kimi/DeepSeek (exhausted credits) produced nothing. This document records what
was verified, what was acted on before the Spider judge runs, what is deferred to climb 3, and what is disputed.

## Verdict on the step-50 claim
Downgraded. "Movement confirmed" → "in-family, in-process movement on 101 BIRD-challenging tasks (11 databases)". Reasons,
all verified: 90 % of the pool and 100 % of the steering eval are BIRD; the steering harness (thinking kept in context,
top_p 0.95, merged tool messages) is not the judge harness (lab.run drops prior thinking, no top_p, one tool message per
call); steps 1–3 on base weights already scored 0.53/0.69/0.70 train pass vs the pool's expected 0.53; the decision rule
"all seeds above the baseline's best seed" is a one-sided test at α ≈ 0.035 per look, looked at twice, and chosen after
seeing the step-25 number. The shift itself is well supported (Opus S2: all 3 step-50 seeds exceed all 11 base seed
measurements, rank p ≈ 0.003) — its SCOPE is the problem, not its existence.

## Acted on before the judge (all in code or bucket now)
1. Pre-registration written (soc 10:35) and amended (soc 11:10): primary = step 150 (archived by the launcher — the loop
   never archives it), same-job base × 8 seeds with pinned vLLM 0.30.0, step150 × 5 seeds, paired per-task estimand,
   30-DB cluster bootstrap primary + McNemar + sign-flip secondary, strata (100 disjoint-schema vs 35 seen-schema tasks;
   55 base-reachable vs 80 never), transition decomposition, prior base as sensitivity only.
2. Per-task eval outcomes logged (a210679) — steering evals from the next VM life on; judge evals always had them.
3. `infra/judge_inprocess.py`: the trainer's rollout path on the Spider 135 for base and step150 (secondary diagnostic that
   separates "no transfer" from "context-format shift").
4. Eval job caps raised (330 min, $14); seeded Spider base moved to `runs-prior-base/`.
5. Wording retired: pass@8 "ceiling"; "explores longer" (turn growth is the predicted signature of mean-normalised loss).

## Verified and deferred to climb 3 (plan/pending.md, "Climb 3 — changes from the cold reviews")
Context parity (thinking in history, top_p, tool-message format); fixed-snapshot reward + a prompt that promises hidden
rows (hardcoding is latent: 0 FROM-less passes in 16k episodes, but a detector is needed); loss normalisation (Dr. GRPO
length bias; the length penalty is inert: 83 % of passes are under the 6,000-char target); temperature-consistent
log-probs on the sampled token ids, no fabricated EOS in the loss; 37 trivially matchable golds (21 empty); BIRD EX
semantics vs Spider column rules (opposite projection habits); schema decontamination (35/135 Spider tasks on seen
schemas); groups with only {wrong, no-submit} count as signal; dynamic sampling has ~no effect at 2.3 draws/task and
never re-measures p̂; regression guard inert for climb 2; pool 90 % BIRD, Spider-shaped 10 %; verifier rejects 8/24 of
Spider's own gold SQL (0.170 is not an official number; ignore_order True on 135/135; condition_cols ⊊ gold on 45/135);
n = 1 training run (2 × 75 steps next time); named large-model comparator in the same harness; 12-seed judge is ~$3.50.

## Disputed or moot
- BIRD "first gold only" (DeepSeek): every BIRD task has exactly one gold.
- Cross-engine TPC gold (DeepSeek): same exported data; 5 hand-translated SQLite golds pass verify_spider.
- "Lucky LoRA init" (DeepSeek): PEFT initialises B to zero (Opus).
- Hardcoding and 5,000-row truncation as BLOCKS (codex): measured prevalence 0 and 8/521 pool, 0/135 Spider (Opus).
- "Dynamic sampling removes the hardest tasks" (codex, DeepSeek): at 2.3 draws per task it barely acts and skews toward
  easy tasks (Opus R5). The design defect is the missing re-measurement, not the rest period.
- +10-point margin (codex, DeepSeek): arbitrary; the design question is power and a comparator (Opus S5). We keep +5 as
  "detected", not "large".
- space-bunny's "primary should be McNemar": the clustering is real (ICC 0.16), so the cluster bootstrap stays primary and
  McNemar is reported beside it.

## What the reviews cost
codex ~$0 (subscription); DeepSeek 51k tokens ≈ $0.15; Opus subagent 232k tokens; space-bunny free tier. About 70 minutes
of wall time in parallel with the climb.

## Addendum, 2026-09-28 00:30 IST: the judge the reviews asked for
The pre-registered analysis (`lab/judge_stats.py`, 30-DB cluster bootstrap primary) ran on the climb-2 checkpoint.
- Spider 2.0-Lite SQLite (target, 135 tasks): step150 × 5 seeds vs base × 8 seeds, 0.141 vs 0.169, Δ −2.87 points,
  CI [−6.16, +0.35], McNemar p 0.12. Success criterion NOT MET. Base-reachable stratum −10.7. Exploratory single-seed
  checkpoint sweep: step80 +4.5 [+0.5, +8.2], step120 −4.4 [−8.9, −0.4].
- BIRD Mini-Dev (in-family secondary, 496 tasks, 11 DBs): step150 × 3 vs base × 3, 0.514 → 0.595, Δ +8.10,
  CI [+4.92, +11.64], McNemar p < 0.001, DB sign-flip p 0.002; base-never stratum 0 → 18.1; no-submit 0.031 → 0.009.
- The in-process diagnostic (Opus C1, context parity) did not run: the script failed at import on the VM. Fixed in
  `infra/judge_inprocess.py`; it needs one more judge life (≈45 min).
Reading: Opus's forecast held. The pool (90 % BIRD) taught BIRD conventions; the gain did not transfer to analytical
SQL, and the late checkpoints moved against the target. The checkpoint sweep is single-seed and unconfirmed; the
morning decision is a 5-seed re-run of steps 60/80/100 plus the diagnostic in one life (≈$3), then climb 3 with
off-family checkpoint selection (TPC-DS held-out dev set) and a TPC-majority pool. Full numbers: leg Experiment 12.
