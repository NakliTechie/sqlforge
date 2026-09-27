# Cold review of sqlforge climb 2 (Claude Opus 5.5 subagent, 2026-09-27 ~10:50–11:05 IST)

Read-only review of reports/review-brief-2026-09-27.md, the code, plan/ and the local run mirrors; numbers recomputed from
runs/ and lab/*.json. Read after codex and DeepSeek. Saved verbatim from the subagent's final message by the caller.

## 0. Two operational defects to fix before the Spider judge runs
1. BLOCKS. Step 150 will never be archived: infra/climb_loop.py archives only n % 20 == 0 and climb2-startup passes
   --save-every 1000; spider2-eval-climb2-startup.sh reads only saves/ → LAST = step140. Copy ckpt/step150 → saves/step150 by hand.
2. BLOCKS. The climb-2 regression stop rule cannot fire: climb_loop.regression() needs exec_acc_by_set["handmade"]; BIRD-
   challenging tasks have no `set` key, so evaluate writes {"all": …}. No automated guard against collapse.

## 1. Confounds
C1 BLOCKS. Steering eval and Spider judge put different context in front of the model: training/steering keep every prior
   turn's <think> block (train/rollout.py:76-85); lab.run --backend openai appends only `content` and drops reasoning
   (harness/loop.py:115-118). Plus top_p 0.95 vs unset, and merged vs per-call tool messages. Observed signature on the same
   101 BIRD-challenging tasks with the base: in-process 10.9 turns / no-submit 0.063 vs lab.run 9.1 / 0.033. Only the
   thinking-retaining path can reach the 32k limit (incident 3). A Spider null cannot separate "no transfer" from "context shift".
C2 DEGRADES (BLOCKS for "unseen schema" claims). Databases are disjoint by name only. Table-name Jaccard: Spider
   sqlite-sakila (7 tasks) + Pagila (2) ≡ pool movie_3 (1.00); IPL (11) ~ pool soccer_2016 (0.38); AdventureWorks (1) ~ pool
   works_cycles; EU_soccer (5) ≡ steering european_football_2 (1.00); f1 (9) ~ steering formula_1. 35 of 135 Spider tasks
   (26 %) sit on schemas seen in training or steering; 21 on training-pool schemas. Climb 1 already differed by stratum:
   +3.8 points on overlap tasks vs +1.0 on the rest.
C3 DEGRADES. Pool mean base p̂ 0.53; train pass 0.645 over steps 1–25 — a 12-point gap explained by fast learning or by the
   in-process harness scoring higher than lab.run (C1). Steps 1–3 rows (base weights) decide; not in the local mirror.
C4 DEGRADES. Loss normalisation favours long failures: sequence_logprobs returns the MEAN log-prob per trajectory (grpo.py:69),
   loss -(adv*lp)/len(batch) → per-token push scales 1/L (Dr. GRPO length bias); std normalisation adds difficulty bias. Base
   BIRD-train failures are longer than passes (median 5,641 vs 3,179 generated chars) → long failures get ~half the penalty →
   rising turns with flat no-submit, as observed. The success-gated length penalty is nearly off: --target-chars 6,000 default,
   83 % of base passes under 6,000 chars.
C5 DEGRADES. The trainer does not score the tokens it sampled: temperature mismatch (as codex/DeepSeek) plus re-tokenization of
   decoded text per segment (rollout.py:149-156) and a fabricated <|im_end|> on truncated 2,048-token turns marked as generated.
C6 DEGRADES. vLLM unpinned in both arms; pre-registration reuses the 2026-09-26 base while the eval script would re-run it →
   two candidate bases.

## 2. Statistics
S1. "All 3 seeds above the baseline's best seed" fires on 22 of 560 three-vs-three splits of the 8 base seeds (3.9 %/look).
S2. The in-family shift on the 101 tasks is better supported than codex's p = 0.163: all three step-50 seeds exceed all 11
   base seed measurements (8 lab.run + 3 in-process); rank probability 1/C(14,3) = 0.0027 (caveat: two harnesses). Pairing by
   seed is invalid. Scope, not existence, is the problem; 11 clusters only.
S3. Per-task outcomes for steps 0/25/50 do not exist; life 9 runs the archive it pulled at boot, so steps 75+ log per task only
   after a relaunch. Re-run step-0 and step-50 evals offline to pair.
S4. Spider seed noise: 8 base seeds pass 26/22/21/19/23/26/23/18 → SD 2.73 tasks, range 8 (not "±5").
S5. Power of the pre-registered 3×3 design at its own +5 margin ≈ 0.61; 8 base × 5 trained → 0.87; 8×8 → 0.94.
S6. Replaying the plan on climb 1: +1.73 points, DB-cluster bootstrap 95 % CI [−1.6, +5.1], sign-flip p 0.41; clustering
   barely widens a paired interval, within-task Bernoulli noise dominates → more seeds is the cheap fix.
S7. n = 1 training run; the in-family shift is visible by step 25–50 → 2 × 75 steps beats 1 × 150 for inference.
Plan for the Spider comparison: freeze step 150 (restored to saves/), same-job base with pinned vLLM and interleaved requests,
lab.run with explicit top_p, identical seeds; y[m,t,s]; Δ = mean paired per-task difference; DB-cluster bootstrap (B 20k,
percentile + BCa) + DB sign-flip test + GLMM check; "beats base" = lower bound > 0, "meaningful" = Δ ≥ +5; ≥ 8 base × 5
trained seeds; pre-declared strata (100 disjoint-schema vs 35 overlap; 55 base-reachable vs 80 never); transition
decomposition (no-submit→pass vs wrong→pass); strict secondary verifier + manual audit of new passes; Holm over checkpoints.

## 3. Reward, verifier and pool design
R1 DEGRADES. 37 of 521 pool tasks have trivially matchable golds: 21 empty gold results (any zero-row query passes; mean base
   p̂ 0.458) and 16 single cells equal to 0/1/NULL. Spider has 1 empty-gold task.
R2 DEGRADES. The BIRD reward is not official BIRD EX: sorted-list (multiset) compare after NULL→0 vs set(pred)==set(gold).
R3 DEGRADES. On the Spider-like distribution the gradient mostly teaches commitment: among base zero-pass tasks, 58/80 Spider
   and 36/44 TPC-DS still show a wrong/no-submit mix (group kept, no passing SQL); TPC-DS pool rollouts are 14 % no-submit vs 2 %.
R4 DEGRADES. BIRD (exact columns, penalises extras) and Spider (ignores extras, requires every gold column; condition_cols empty
   on 86/135) reward opposite projection habits; Spider ignore_order on all 135 vs 40/52 TPC training tasks requiring order.
R5 COSMETIC. Dynamic sampling barely acts: each task is drawn ~2.3 times in 150 steps; resting skews toward easy tasks.
R6 latent. Hardcoding: 0 passing submissions without FROM in >16,000 episodes; needs a perturbed-DB detector.
R7 COSMETIC. 5,000-row truncation: 8/521 pool golds, 0/135 Spider.  R8 COSMETIC. TPC-H: 27 tasks from 18 public queries.

## 4. What Spider can and cannot establish
Can: whether the frozen step-150 adapter beats base on these 135 tasks under lab.run, with a DB-clustered interval, once §0 is
fixed. Cannot: "large-model quality" (no comparator in this harness); transfer to unseen schemas unless it holds on the 100
disjoint tasks; new capability unless gains appear on the 80 never-pass tasks; RL as cause (n = 1, no controls); a null as
"no transfer" (C1 — needs a diagnostic evaluate() run on Spider); a pristine test set (2,025 episodes already; informed the
pool). Convincing: Δ ≥ +5, lower bound > 0 at 8 × ≥5 seeds, same sign on the disjoint stratum and DB-weighted mean, gain
mostly wrong→pass, strict verifier agreeing, second training run within the CI.

## 5. Top-3 changes for the next climb
1. One rollout object for measurement, training, steering, judging (same context rule, explicit sampling params, score the
   returned token ids with temperature-scaled logits; selfcheck asserting identical prompt token ids). Re-run the base.
2. Fix loss/reward pressures: constant or token-sum normalisation, drop std scaling, a turn cost on failures, train only groups
   with ≥1 pass or no-submit ≈ −0.25, remove the 37 trivial golds, official BIRD set semantics or strict one-to-one columns;
   log per step groups-with-a-pass, entropy, generated tokens for passes vs fails, overflow count.
3. Steer on the target, decontaminate by schema (drop sakila/IPL/AdventureWorks schemas from the pool or pre-declare the
   stratum), replicate as 2 × 75 steps.

## 6. Where I disagree with the earlier reviews
Codex's seed t-test understates the in-family evidence (S2). "Dynamic sampling removes the hardest tasks" ignores draw frequency
(R5). 5,000-row truncation and hardcoding are not BLOCKS (R6, R7). +10 points is arbitrary; power and a comparator are the
issues. "Lucky LoRA init" — PEFT inits B to zero. "Length variance among passes never trains" — mixed groups use shaped rewards;
the real defect is the inactive penalty (C4). "BIRD first gold only" — moot. "Context guard teaches avoidance" — the cause is
the context format (C1).

## 7. What both earlier reviews missed
Step 150 never archived; regression guard inert; judge drops prior thinking; 35 Spider tasks on seen schemas; the 12-point
train-pass gap; length-normalised loss explains turn growth and the penalty is off; re-tokenization + fabricated EOS in the loss;
unpinned vLLM and two bases; 37 trivial golds; non-official BIRD EX; BIRD vs Spider projection conflict; ~60 % power; per-task
logs missing for steps 0–50.

## 8. Things the brief got wrong
"One turn rule shared" (parse rule only); "Spider untouched" (2,025 episodes, informed pool design); "DBs disjoint" (by name);
"groups with one outcome class dropped" (wrong/no-submit mixes kept); "success-gated length penalty" (inactive for most passes);
"BIRD exact-row-set EX" (multiset + NULL→0); "±5 tasks" (range 8); "step-150 × 3" (not archived; under-powered); "explores
longer" (predicted signature of length-normalised loss; no transcript evidence).

Verdict on the nine worries: 1 real · 2 real as a rule, shift supported (S2) · 3 real, mechanism C4 · 4 real, R1 measurable ·
5 partly, R4 larger · 6 real, C4/C5 > clipping · 7 minor for climb 2 · 8 real for climb 3 · 9 answered in §2/§4.
