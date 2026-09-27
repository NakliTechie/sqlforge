No. The step-50 result does not mean what you say it means. It is a BIRD-family adaptation signal, not a Spider result, and the statistical rule you used to call it “confirmed” is not sound.

## 1. Flaws/confounds that make the step-50 result not mean what you think

**The pool and the steering eval are the same family; the target is a different family.**  
469 of 521 pool tasks are BIRD-train (`lab/climb2_pool.json`); the steering eval is BIRD Mini-Dev challenging (101 tasks, 3 seeds). The final judge is Spider 2.0-Lite SQLite (135 tasks). TPC-DS is the only measured Spider-twin — Exp. 10 says pass@1 0.170 vs 0.165, pass@8 0.389 vs 0.407, never-pass 61% vs 59%, turns 21 vs 19, 500 vs 445 s/episode — but TPC-DS is only 25/521 = 4.8% of the pool. TPC-H is 27/521 = 5.2%. So 90% of training is on BIRD, and the steering eval is BIRD. The step-50 move 0.406 → 0.495 (+8.9 points) is therefore in-family movement. It does not establish transfer to Spider.

**BIRD label noise is a first-order confound, not a footnote.**  
You state BIRD has a documented ~50% annotation-error rate. The band filter (`lab/pool_from_passk.py`, 0 < p̂ < 1) removes never-pass tasks, but it does not remove wrong golds that the model sometimes matches by accident. RL will reinforce whatever SQL matches the wrong gold. BIRD Mini-Dev shares annotators and the evidence-hint convention. The +9 tasks on 101 BIRD-challenging tasks can easily be partly fitting annotation errors or BIRD style. Spider does not share those errors.

**One training run, three eval seeds.**  
You have one climb-2 training trajectory. The 3 seeds are eval sampling seeds, not training seeds. The step-50 improvement could be a lucky LoRA init, data order, or vLLM nondeterminism. Step-0 seed spread was 10.9 points (0.337 / 0.436 / 0.446). Step-50 mean is 0.495 vs step-0 mean 0.406. One seed drives 63% of the average paired difference: seed 0 is +16.8, seeds 1 and 2 are +6.9 and +3.0. That is fragile.

**Checkpoint/stopping selection uses the proxy.**  
The pre-reg says Spider is the final judge, but you are steering and stopping by BIRD. If you later pick the best BIRD checkpoint and evaluate it on Spider, Spider is no longer a clean judge. You are selecting on a proxy that is 90% the same family as the training pool. The final Spider must be run on a pre-registered step or on all checkpoints with multiple-comparison correction.

**The context guard changes the training distribution.**  
`train/rollout.py:run_episodes` now ends an episode as a turn-cap if `prompt + max_new_tokens > max_model_len`. Incident 3 says one prompt reached 32,769 tokens. That episode gets reward −1 as a no-submit. This is a harness-induced −1, not a model choice. If the measurement harness (`harness/loop.py`) does not have the same guard, the trained object and measured object differ. This is a distribution shift, especially on long real-schema tasks.

**Cross-engine gold for TPC-H/TPC-DS is a label-noise source.**  
The brief says “Golds for TPC-H/TPC-DS come from DuckDB on the same generated data exported to SQLite.” But TPC-H/TPC-DS tasks run on SQLite (`train/run_train.py:task_env`, kind == “spider2”). If the gold result is computed by DuckDB and the model SQL is executed on SQLite, correct SQLite SQL can fail due to dialect differences. TPC-DS is your closest Spider-twin, so this matters.

## 2. Statistical weaknesses

**“All seeds above the baseline’s best seed” is not a sound rule.**  
Baseline best seed 0.446 is an order statistic of 3 seeds. Step-50 seeds are 0.505 / 0.505 / 0.475. The correct comparison is mean 0.495 vs mean 0.406, +8.9 points. But with 101 tasks, the per-seed binomial SE is about 5 points. Three seeds is not enough for the inference you are making. You need a per-task paired analysis: for each task, pass count out of 3 for step-0 and step-50, then a paired bootstrap, Wilcoxon, or McNemar. You report “paired by seed,” but that is not paired by task.

**Multiple looks inflate alpha.**  
You looked at step 25 and step 50. You will look at steps 75/100/125/150. If “step 50 decides” was chosen after seeing step 25, the reported effect is biased. Pre-register step 150 Spider as the primary endpoint, or use sequential alpha spending.

**The band filter is noisy.**  
`lab/pool_from_passk.py` uses base pass@8. With 8 samples, p̂ = 1/8 or 2/8 is very noisy. Selecting 0 < p̂ < 1 favors tasks that were lucky enough to pass at least once. Re-measure p̂ under the current policy, or use at least 16–32 samples for band selection.

**Dynamic sampling systematically removes low-p learnable tasks.**  
`train/run_train.py` drops a task after 2 zero-variance groups = 16 rollouts. For true p = 0.1, P(0 passes in 16) = 0.185. For p = 0.05, P = 0.44. So it drops exactly the hardest tasks that might still be learnable. Re-admit after 25 steps is too late. This biases the active pool easier and can inflate train pass .65–.68.

**No training-seed replication.**  
All claims are n = 1 training run. Eval seeds do not address training variance. At minimum, you need 2–3 training seeds to claim the method works. Otherwise the step-50 result is anecdotal.

**Turns and wall time rose; no length control.**  
Turns 10.9 → 13.5, eval wall 836 → 1,720 s. The success-gated length penalty (`train/grpo.py:shaped_reward`) only applies to passes. If all rollouts pass but vary in length, `outcome_class` returns all 1s and the group is dropped (`train/grpo.py:outcome_class`). So length variance among passes never trains. There is no penalty for long failures. The model can stall on failures until the budget nudge, then submit wrong (0) instead of no-submit (−1). That is a cost regression, not evidence of exploration.

## 3. Reward/verifier/pool design errors

**No-submit = −1, wrong = 0, pass = 1 is a reward-hacking surface.**  
The gap no-submit → wrong is the same size as wrong → pass. This strongly incentivizes submitting anything. Climb 1 is the signature: Spider no-submit 0.33 → 0.18 at equal turns, but exec acc 0.170 → 0.188 inside seed noise. The model learned to commit, not to solve. In climb 2, no-submit is flat (.063 → .066), but turns rose. You are not seeing more submission; you are seeing more exploration/stalling.

**Verifier leniency is real.**  
`lab/verify.py:_cell_eq` (lines 125–135) uses `abs_tol=1e-2` and NULL→0. `lab/spider2.py:verify_spider` accepts extra predicted columns because `verify.compare` checks each gold column against *any* predicted column vector; extra columns are ignored. Row count must match, but extra columns are allowed. BIRD EX only uses `task["gold_results"][0]` and normalizes NULL→0. For training, use a stricter verifier: exact columns, strict NULL, no extra columns, gold executed on the same engine. Keep official rules only for the final Spider judge.

**Pool composition is wrong for the thesis.**  
The target is Spider 2.0-Lite analytical multi-step SQL. TPC-DS is the measured twin, but it is 4.8% of the pool. BIRD is 90% and has ~50% annotation error. This is not a pool built at the target distribution. It is a BIRD pool with a Spider judge.

**Dynamic sampling removes the hardest tasks.**  
As above, dropping after 2 zero-variance groups removes low-p tasks. The code also clears all dropped tasks if `active < tasks_per_step`, but with 521 tasks and 8 per step, that almost never happens. The active pool shrinks and gets easier. Monitor the p̂ distribution of active tasks; if it shifts up, the climb is on easier data.

**GRPO specifics are not clean.**  
`train/grpo.py:grpo_step` says “on-policy, one update per batch, so the importance ratio is 1 and the clipped objective reduces to -A * logp.” But generation uses T=0.6 and top_p=0.95, while `sequence_logprobs` computes logprobs from the raw softmax. The sampling policy is not the raw policy. That gradient is biased. There is no KL/reference term, no clipping ratio, lr 1e-5 constant, LoRA r32. With ~5.0–5.3 kept groups per step out of 8 (about 35% dropped), effective batch is ~42 trajectories. This is high variance and unstable. Add KL or clipping, or correct for temperature/top-p.

**BIRD EX only uses the first gold.**  
In `lab/spider2.py:verify_spider`, the `bird_ex` branch uses `gold = task["gold_results"][0]`. If BIRD tasks can have multiple valid gold SQLs, correct alternatives are marked wrong. Spider loops over all `gold_results`; BIRD does not. This can drop learnable tasks or give false-negative rewards.

**Context guard adds harness-induced −1s.**  
`train/rollout.py:run_episodes` ends episodes on context overflow as a turn-cap. That reward is −1. The model did not choose to stop. This can teach avoidance of long-context real-schema tasks, which are exactly the Spider-like tasks.

## 4. What the Spider result at the end can and cannot establish

**Can establish:**  
On this specific 135-task Spider 2.0-Lite SQLite slice, under this exact agentic harness, this checkpoint from this one training run achieved X% exec accuracy vs base Y%. If you use a pre-registered paired task-level test, it can support that the training procedure improved this slice. It can also show behavioral changes: no-submit rate, turns, hit-cap rate.

**Cannot establish:**  
- “Large-model quality on multi-step analytical SQL.” 135 tasks, one dialect (SQLite), one harness. The reference GPT-4o 15.6 is not the same harness. Base Qwen3.5-4B is already 0.165 on this slice, above that reference. You need a large-model baseline in the same harness.
- That RL/GRPO caused the improvement. No training-seed replication, no control for pool composition, reward changes, or verifier leniency.
- That the model learned new capability rather than BIRD style/annotation-error adaptation. The pool is 90% BIRD; BIRD Mini-Dev shares annotators and evidence hints. Spider does not.
- That hard tasks were learned. The pool excludes never-pass tasks (0 < p̂ < 1). Spider has 80/135 never-pass tasks. Any direct training signal is on tasks the base sometimes solves. The pass@8 ceiling for Spider is 55/135 = 0.407. Generalization to never-pass tasks is untested.
- That BIRD steering is predictive of Spider. TPC-DS is the measured twin; BIRD is not.
- That verifier leniency did not inflate gains. Official Spider rules accept extra columns, NULL→0, abs_tol 1e-2. Training on BIRD with lenient EX can teach hacks that transfer poorly.
- That checkpoint selection is unbiased. If you pick the best BIRD checkpoint and evaluate it on Spider, Spider is selected. Pre-register the step(s).

**What a convincing final Spider result looks like:**  
Pre-register step 150 (or a fixed rule) on the 135 tasks. Use at least 5 eval seeds, not 3. Report per-task pass counts out of seeds and use a paired task-level bootstrap/McNemar. Require a mean improvement of at least +10 points (13.5 tasks) with a 95% CI lower bound > 0, and no seed regression. Run a large model (GPT-4o or comparable) in the same harness. Report pass@8, no-submit, turns, hit-cap. If you get +2 points like climb 1 (+1.8), it is not supported.

## 5. Three highest-value changes for the next climb

1. **Rebalance the pool to the target distribution.**  
   TPC-DS is your measured Spider-twin (Exp. 10), but it is only 25/521 tasks. Make TPC-DS/Spider-like analytical tasks the majority (>70%). Author more TPC-DS questions or generate parameter variants. Use a held-out TPC-DS dev set for steering, not BIRD. Cap BIRD-train at <20–30%, or drop it until its ~50% annotation error is audited and filtered. This directly addresses the transfer confound.

2. **Fix reward/verifier and dynamic sampling to prevent hacking and noise.**  
   Use a strict training verifier: exact columns, strict NULL, no extra columns, gold executed on the same engine. Change no-submit reward from −1 to something that does not dominate wrong (e.g., −0.1 or 0), or add a per-turn cost. Apply length penalty to all trajectories, or drop it and use a turn cap. Add KL/clipping or correct for T=0.6/top-p. For dynamic sampling, do not drop low-p tasks after 2 zero-variance groups (16 rollouts); use forced exploration or re-measure p̂ under the current policy with more samples. This prevents the model from learning to submit garbage/stall and keeps the hard learnable tasks.

3. **Redesign statistics and checkpoint selection.**  
   Pre-register the primary endpoint (step-150 Spider) and stopping rule. Use ≥5 eval seeds and paired task-level tests with CIs. Replicate training with ≥2 seeds if budget allows. Run a large-model baseline in the same harness. Report all checkpoints, not just the best BIRD one, with multiple-comparison correction. This makes the final claim credible.

Your nine worries: 1–7 are real, 8 is real and you should re-measure p̂ under the current policy, 9 is underpowered as currently designed. The biggest things you missed: one training run, 90% BIRD pool with ~50% label noise, TPC-DS only 4.8% of the pool, context-guard-induced −1s, cross-engine TPC-H/DS gold, BIRD EX using only the first gold, and no large-model baseline in the same harness.


<!-- usage: {"prompt_tokens": 21077, "completion_tokens": 30016, "total_tokens": 51093, "prompt_tokens_details": {"cached_tokens": 20864}, "completion_tokens_details": {"reasoning_tokens": 26580}, "prompt_cache_hit_tokens": 20864, "prompt_cache_miss_tokens": 213} -->
