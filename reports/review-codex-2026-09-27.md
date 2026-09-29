## Confounds

[RISK inferred]

**BLOCKS.** Step 50 measures BIRD-family adaptation, not general SQL transfer. BIRD supplies 469 of 521 training tasks and all 101 steering tasks.

The database split prevents schema memorization. It does not separate annotation style, evidence hints, or task construction conventions.

evidence: [review brief:48-63](<repo>/reports/review-brief-2026-09-27.md:48), [BIRD prompt construction:129-149](<repo>/lab/spider2.py:129)

mitigation: Add a contemporaneous TPC-DS or withheld non-BIRD steering set. Never select checkpoints solely through BIRD Mini-Dev.

[RISK inferred]

**BLOCKS.** The training and external measurement harnesses are not equivalent.

Training fixes `top_p=0.95`. The external harness omits `top_p`, leaving it to the server configuration.

Training combines multiple tool results into one message. The external OpenAI harness returns one tool message per call.

Only the training runner applies the 32K context guard. These differences can alter trajectories despite a shared turn parser.

evidence: [train/vllm_policy.py:36-64](<repo>/train/vllm_policy.py:36), [train/rollout.py:89-125](<repo>/train/rollout.py:89), [harness/loop.py:61-75](<repo>/harness/loop.py:61), [harness/loop.py:145-152](<repo>/harness/loop.py:145)

mitigation: Use one rollout implementation and one explicit sampling configuration for pool measurement, training evaluation, and final evaluation.

[RISK inferred]

**BLOCKS.** The step-50 comparison crosses a code change.

The first step-50 evaluation crashed on a 32,769-token prompt. The rerun converted overflows into no-submissions after restoring step 40.

The repository also states that batched requests need not reproduce serial outputs. Changed batch membership can therefore change later generations.

evidence: [plan/soc.md:3-5](<repo>/plan/soc.md:3), [train/rollout.py:114-131](<repo>/train/rollout.py:114), [lab/run.py:47-49](<repo>/lab/run.py:47)

mitigation: Rerun step 0, step 25, and step 50 under the final code and identical batching.

[RISK inferred]

**DEGRADES.** The run has no independent training replicate. One initialization, sampling stream, and task order produced the entire curve.

Three evaluation seeds quantify rollout variation around one training run. They do not quantify training instability.

evidence: [plan leg:397-402](<repo>/plan/lab/task-synth/2026-09-24-leg.md:397), [train/run_train.py:124-135](<repo>/train/run_train.py:124)

[RISK inferred]

**BLOCKS.** Real-schema training uses one fixed database per task. The system prompt falsely promises evaluation on different rows.

A policy can inspect values through `run_sql`, then submit database-specific predicates or constants. That behavior receives full reward.

evidence: [harness/system.md:3-7](<repo>/harness/system.md:3), [lab/spider2.py:59-71](<repo>/lab/spider2.py:59), [train/run_train.py:47-64](<repo>/train/run_train.py:47)

mitigation: Train against hidden database variants. Otherwise, change the prompt and classify the reward as snapshot accuracy.

## Statistics

[RISK inferred]

**BLOCKS.** “Every seed beats the baseline’s best seed” is not a statistical test. The baseline maximum is a selected order statistic.

The paired seed differences are 17, 7, and 3 tasks. Their mean is 9 tasks with standard deviation 7.21.

Treating three seeds as units gives \(t=2.16\), two-sided \(p=0.163\). Its illustrative 95% interval spans about −9 to +27 tasks.

That calculation is still invalid for inference because seeds are not the population unit.

evidence: [review brief:53-57](<repo>/reports/review-brief-2026-09-27.md:53), counts are 34/44/45 versus 51/51/48 from the reported rates.

mitigation: Use paired task outcomes and cluster resampling by database.

[RISK inferred]

**BLOCKS.** The 303 evaluations contain 101 repeated tasks from only 11 databases. They are not 303 independent benchmark items.

The largest database contributes 18 tasks. Database-specific difficulty therefore affects the aggregate.

evidence: read-only count from [lab/bird_challenging.json](<repo>/lab/bird_challenging.json): 101 tasks across 11 schemas, with 18 from `toxicology`.

mitigation: Report database-cluster bootstrap intervals and per-database changes. Also report task-level paired wins, losses, and ties.

[RISK inferred]

**DEGRADES.** The steering set is inspected every 25 steps. Continued checkpoint selection converts it into development data.

A monotone three-point curve does not correct this repeated-selection bias. More evaluations increase the chance of selecting favorable noise.

evidence: [review brief:50-57](<repo>/reports/review-brief-2026-09-27.md:50), [plan leg:409-419](<repo>/plan/lab/task-synth/2026-09-24-leg.md:409)

[RISK inferred]

**DEGRADES.** The pool filter has severe selection noise. Eight samples only permit estimates in increments of 0.125.

My read-only count finds 122 of 521 tasks at 7/8. Another 79 sit at 1/8.

Thus, 201 tasks sit one outcome from exclusion. Their measured difficulty contains substantial winner’s-curse bias.

evidence: [pool builder:34-43](<repo>/lab/pool_from_passk.py:34), read-only counts from [lab/climb2_pool.json](<repo>/lab/climb2_pool.json)

[RISK inferred]

**BLOCKS.** The final Spider plan tests base plus every twentieth checkpoint. Selecting the best checkpoint afterward invalidates the “untouched judge” description.

Only a checkpoint fixed before opening Spider can support a primary claim. Other checkpoint results must remain exploratory.

evidence: [review brief:50-51](<repo>/reports/review-brief-2026-09-27.md:50)

mitigation: Declare step 150 as the sole primary checkpoint before evaluation. Apply multiplicity correction to any checkpoint sweep.

## Reward/verifier/pool design

[RISK inferred]

**BLOCKS.** The reward makes arbitrary submission as valuable as correctness improvement.

The gap from no-submit to wrong-submit is 1. The nominal gap from wrong-submit to a correct submission is also 1.

Length shaping can reduce the second gap below 1. This reward therefore trains commitment at least as strongly as correctness.

evidence: [train/grpo.py:20-31](<repo>/train/grpo.py:20), [climb-1 behavior:270-278](<repo>/plan/lab/task-synth/2026-09-24-leg.md:270)

mitigation: Reduce the no-submit penalty. Measure correct, wrong-submit, and no-submit transitions separately.

[RISK inferred]

**DEGRADES.** Mixed groups containing only wrong submissions and no-submissions still produce gradients. Those groups contain no evidence about correct SQL.

The dynamic sampler also regards them as informative because `outcome_class` preserves −1 versus 0.

evidence: [train/grpo.py:34-46](<repo>/train/grpo.py:34), [train/run_train.py:257-281](<repo>/train/run_train.py:257)

[RISK inferred]

**DEGRADES.** Dynamic resting conflates solved tasks with unsolved tasks. Two consecutive 8-rollout constant groups trigger the same 25-step exclusion.

Re-admission resets the streak without estimating current-policy pass rate. It does not return the task to the learnable band.

evidence: [train/run_train.py:244-276](<repo>/train/run_train.py:244), [review brief:73-76](<repo>/reports/review-brief-2026-09-27.md:73)

[RISK inferred]

**DEGRADES.** The update lacks KL and entropy control. It also uses a constant learning rate and full-trajectory scalar advantage.

The objective averages token log-probability within each trajectory. It does not diagnose policy collapse, repetition, or excessive exploration.

evidence: [train/grpo.py:1-8](<repo>/train/grpo.py:1), [train/grpo.py:54-84](<repo>/train/grpo.py:54), [train/run_train.py:116-146](<repo>/train/run_train.py:116)

[RISK inferred]

**BLOCKS.** The BIRD verifier truncates both predicted results and generated gold results at 5,000 rows without rejecting overflow.

A query can therefore pass by matching only an arbitrary 5,000-row prefix. The builder creates BIRD golds through that same truncating function.

evidence: [lab/spider2.py:39-56](<repo>/lab/spider2.py:39), [lab/spider2.py:129-149](<repo>/lab/spider2.py:129)

mitigation: Reject results exceeding the cap. Alternatively, stream and hash the complete multiset.

[RISK inferred]

**BLOCKS.** Spider matching accepts extra predicted columns. It also permits one predicted vector to satisfy multiple gold columns.

`NULL` becomes zero before comparison. Numeric equality uses a 0.01 absolute tolerance.

These official rules measure benchmark compatibility. They do not establish semantic SQL equivalence.

evidence: [lab/verify.py:125-160](<repo>/lab/verify.py:125), [lab/spider2.py:59-90](<repo>/lab/spider2.py:59)

mitigation: Add strict column cardinality and one-to-one matching as a secondary metric. Audit all newly passing tasks manually.

[RISK inferred]

**DEGRADES.** The training pool is 90.0% BIRD by task count. TPC-DS and TPC-H jointly contribute only 52 of 521 tasks.

The climb therefore emphasizes BIRD conventions rather than the stated analytical-SQL target.

evidence: [review brief:48-49](<repo>/reports/review-brief-2026-09-27.md:48), read-only count from [lab/climb2_pool.json](<repo>/lab/climb2_pool.json)

## What Spider can and cannot establish

[RESULT inferred]

A predeclared step-150 comparison can establish better expected execution-match accuracy on the 135-task local SQLite slice under this harness.

That claim requires paired task-level analysis with database clustering. Its interval must exclude zero.

evidence: [review brief:9-12](<repo>/reports/review-brief-2026-09-27.md:9), [review brief:77-78](<repo>/reports/review-brief-2026-09-27.md:77)

[RISK inferred]

**BLOCKS.** Spider cannot establish “large-model quality.” The success criterion only requires beating the 4B base by any positive amount.

The brief specifies no large-model comparator, minimum effect, or uncertainty threshold.

evidence: [review brief:9-12](<repo>/reports/review-brief-2026-09-27.md:9)

mitigation: Evaluate a named large model in the identical harness. Predeclare a nontrivial equivalence or superiority margin.

[RISK inferred]

**DEGRADES.** Spider cannot establish semantic correctness beyond one database snapshot. Single-database execution can reward accidental equivalence.

The local validation already records only 16 passing gold queries among 24 available gold-SQL tasks.

evidence: [lab/spider2.py:1-6](<repo>/lab/spider2.py:1), [plan/soc.md:27](<repo>/plan/soc.md:27), [prior-art report:50-54](<repo>/reports/RL%20for%20agentic%20text%20to%20SQL.md:50)

[RISK inferred]

**DEGRADES.** Spider cannot establish performance on full Spider 2.0-Lite, Snow, DBT, BIRD, or unseen production databases.

The evaluation covers 135 local SQLite tasks from 30 databases.

evidence: [review brief:9-12](<repo>/reports/review-brief-2026-09-27.md:9), [lab/spider2.py:1-12](<repo>/lab/spider2.py:1)

[RISK inferred]

**BLOCKS.** A convincing primary result needs a predeclared point margin and an uncertainty gate.

I would require at least +10 percentage points and a database-clustered 95% interval above zero.

I would also require three independent training runs. Evaluation seeds cannot replace training replication.

evidence: the current base is 0.165, while reported rollout variation spans five tasks per seed at [review brief:77-78](<repo>/reports/review-brief-2026-09-27.md:77).

mitigation: Freeze this criterion before reading Spider checkpoint results.

## Top-3 changes for the next climb

[PLAN inferred]

1. Freeze one primary checkpoint and one evaluation implementation.

Run base and adapted policies on identical paired seeds. Use database-cluster bootstrap intervals and three independent training runs.

verifier: The preregistration names the checkpoint, seeds, margin, clustering unit, and multiplicity treatment before evaluation.

evidence: [current multi-checkpoint plan:50-51](<repo>/reports/review-brief-2026-09-27.md:50), [harness mismatches cited above](<repo>/train/vllm_policy.py:36)

[PLAN inferred]

2. Remove single-snapshot reward shortcuts.

Use hidden database variants, strict one-to-one columns, complete-result hashing, and audited nonempty golds. Keep official scoring only as the headline benchmark metric.

verifier: Every rewarded query passes at least three hidden instances and a strict secondary comparator.

evidence: [fixed-database verifier:59-90](<repo>/lab/spider2.py:59), [5,000-row truncation:39-56](<repo>/lab/spider2.py:39), [system-prompt mismatch:5](<repo>/harness/system.md:5)

[PLAN inferred]

3. Replace stale band filtering with current-policy frontier sampling.

Retain groups with two through six successes. Separate wrong-submit gradients from correctness gradients. Add entropy, KL, and turn-distribution monitoring.

Run isolated reward and horizon ablations before another 150-step climb.

verifier: Each arm uses the same tasks, rollout budget, evaluation seeds, and stopping rule.

evidence: [current resting logic:244-281](<repo>/train/run_train.py:244), [current reward:20-46](<repo>/train/grpo.py:20), [prior-art recommendation:101-117](<repo>/reports/RL%20for%20agentic%20text%20to%20SQL.md:101)

## Things the brief got wrong

[RISK inferred]

**BLOCKS.** The step-50 movement label overstates the evidence. Three favorable seed aggregates do not establish a population improvement.

The illustrative paired-seed test gives \(p=0.163\). The proper database-clustered task analysis remains absent.

evidence: [review brief:53-57](<repo>/reports/review-brief-2026-09-27.md:53)

[RISK inferred]

**DEGRADES.** “The policy explores longer before committing” is an interpretation, not an observation.

The observations are 10.9 to 13.5 turns, flat no-submit, doubled wall time, and one context-overflow crash.

These numbers also fit verbosity, repeated failed queries, or delayed submission.

evidence: [plan leg:415-419](<repo>/plan/lab/task-synth/2026-09-24-leg.md:415), [plan/soc.md:5](<repo>/plan/soc.md:5)

[RISK inferred]

**DEGRADES.** “TPC-DS matches Spider almost exactly” compares marginal aggregates only.

TPC-DS contributes 72 tasks from one generated benchmark database. Spider contains 135 tasks across 30 databases.

Matching pass rate and turns does not establish matching SQL structures, language, schemas, or failure modes.

evidence: [review brief:37-45](<repo>/reports/review-brief-2026-09-27.md:37), [plan leg:367-376](<repo>/plan/lab/task-synth/2026-09-24-leg.md:367)

[RISK inferred]

**BLOCKS.** Pass@8 is not a ceiling. It is an eight-sample oracle-union statistic for the base policy.

A 0/8 task can have nonzero base success probability. Training can also create behavior absent from all eight base samples.

Calling 55 reachable tasks a ceiling confuses observed support with a trained-policy limit.

evidence: [plan leg:330-347](<repo>/plan/lab/task-synth/2026-09-24-leg.md:330)

mitigation: Report pass@8 as a pool diagnostic only. Never use it as an attainable-accuracy bound.

[RISK inferred]

**DEGRADES.** The brief treats approximately 50% BIRD annotation error as transferable to this training subset.

The cited audit concerns 52.8% of Mini-Dev. A separate audit found 61% of a sampled BIRD-train subset needed correction.

Neither number identifies which selected tasks have wrong golds. A 0/8 outcome also does not diagnose annotation error.

evidence: [prior-art report:30-32](<repo>/reports/RL%20for%20agentic%20text%20to%20SQL.md:30), [review brief:41-45](<repo>/reports/review-brief-2026-09-27.md:41)

[RISK inferred]

**DEGRADES.** Missing clipping is not the primary GRPO defect here.

The code performs one on-policy update, where the initial importance ratio equals one. The stated reduction is internally coherent.

The material risks are unrestricted drift, stale sampling, fixed-snapshot rewards, and unmonitored entropy.

evidence: [train/grpo.py:72-84](<repo>/train/grpo.py:72), [review brief:71-76](<repo>/reports/review-brief-2026-09-27.md:71)

[RISK inferred]

**BLOCKS.** “Training signal” does not mean useful learning signal.

Five retained groups per step only show reward-class variance. Wrong-submit versus no-submit groups qualify despite containing no successful SQL.

The brief therefore mistakes gradient availability for evidence of capability improvement.

evidence: [review brief:52](<repo>/reports/review-brief-2026-09-27.md:52), [train/run_train.py:257-281](<repo>/train/run_train.py:257), [train/grpo.py:34-46](<repo>/train/grpo.py:34)