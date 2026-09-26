"""One GRPO climb over the synthesised task pool.

uv run python -m train.run_train --model Qwen/Qwen3.5-0.8B --tasks lab/synth_tasks.json \
    --steps 1 --tasks-per-step 1 --group 2 --max-turns 6 --device mps --tag smoke

Every step: sample B tasks → G rollouts each with the current LoRA policy → verify (hidden snapshots)
→ shaped reward → group advantages → one optimizer step. Logs a JSON line per step to runs/train-<tag>.jsonl;
saves the adapter every --save-every steps.

Held-out eval: at step 0 (baseline), every --eval-every steps and at the last step, the current policy runs
the eval tasks once per seed (--eval-seeds, default 1) and the exec_acc / by-hops / cap-rate line goes to
runs/train-<tag>-eval.jsonl. --eval-every 0 disables.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab import schemas, spider2, tasks as fixed_tasks, verify
from train.grpo import group_advantages, grpo_step, outcome_class, shaped_reward
from train.rollout import Episode, HFPolicy, run_episodes, tokenize_trajectory

HARNESS = Path(__file__).parent.parent / "harness"


def make_run_sql(schema):
    con = schemas.build(schema, fixed_tasks.VISIBLE_SEED)

    def run_sql(sql):
        try:
            cols, rows = verify.execute(con, sql)
            return cols, rows, None
        except verify.ExecError as e:
            return [], [], str(e)
    return run_sql


def task_env(task, system_tpl, tools, db_dir):
    """(system prompt, tools, run_sql, verify_fn) for one task. Synth tasks run on generated DuckDB snapshots and verify
    through the G0–G4 ladder with hidden seeds. kind == "spider2" tasks (Spider 2.0, BIRD, TPC-H files; lab/spider2.py)
    run read-only on their SQLite file, the prompt names SQLite, and verify by result match. --db-dir relocates db_path
    by schema name, as lab.run does (VM paths differ from the laptop's)."""
    if task.get("kind") == "spider2":
        path = task["db_path"]
        if db_dir:
            c = list(Path(db_dir).glob(f"{task['schema']}.sqlite")) + list(Path(db_dir).glob(f"**/{task['schema']}/{task['schema']}.sqlite"))
            path = str(c[0]) if c else path
        return (system_tpl.replace("{ddl}", task["ddl"].strip()).replace("DuckDB", "SQLite"),
                json.loads(json.dumps(tools).replace("DuckDB", "SQLite")), spider2.make_run_sql(path),
                lambda sql: spider2.verify_spider(task, sql))
    return (system_tpl.replace("{ddl}", task["ddl"].strip()), tools, make_run_sql(task["schema"]),
            lambda sql: verify.verify(task, sql, fixed_tasks.HIDDEN_SEEDS))


def evaluate(policy, tok, system_tpl, tools, eval_tasks, *, max_turns: int, seeds: list[int], think: bool, db_dir: str = "") -> dict:
    """One rollout per held-out task per seed with the current policy, everything in one lockstep batch; runner-style
    summary plus per-seed accuracy (single-seed numbers on 100–135 tasks span ±5 points — leg Experiments 8–9)."""
    torch.manual_seed(seeds[0])
    envs = [task_env(t, system_tpl, tools, db_dir) for t in eval_tasks]
    eps, meta = [], []
    for seed in seeds:
        for i, (t, (system, tls, run_sql, _)) in enumerate(zip(eval_tasks, envs)):
            eps.append(Episode(tok, system, t["q"], tls, run_sql, max_turns=max_turns, think=think, seed=seed * 10_000 + i))
            meta.append((t, envs[i][3], seed))
    rows = []
    for (task, verify_fn, seed), tr in zip(meta, run_episodes(policy, eps)):
        res = verify_fn(tr.submitted_sql) if tr.submitted_sql else {"pass": False}
        rows.append({"hops": task["hops"], "set": task.get("set", "all"), "seed": seed, "pass": bool(res["pass"]),
                     "hit_cap": tr.hit_cap, "turns": tr.turns, "submitted": tr.submitted_sql is not None})
    n = len(rows)
    hops = sorted({r["hops"] for r in rows})
    sets = sorted({r["set"] for r in rows})
    return {"exec_acc": round(sum(r["pass"] for r in rows) / n, 4),
            "exec_acc_by_seed": {s: round(sum(r["pass"] for r in rows if r["seed"] == s) / len(eval_tasks), 4) for s in seeds},
            "exec_acc_by_set": {k: round(sum(r["pass"] for r in rows if r["set"] == k) / sum(1 for r in rows if r["set"] == k), 3) for k in sets},
            "no_submit_rate": round(sum(not r["submitted"] for r in rows) / n, 4),
            "exec_acc_by_hops": {h: round(sum(r["pass"] for r in rows if r["hops"] == h) / sum(1 for r in rows if r["hops"] == h), 3) for h in hops},
            "turn_cap_rate": round(sum(r["hit_cap"] for r in rows) / n, 4),
            "mean_turns": round(sum(r["turns"] for r in rows) / n, 2), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tasks", default="lab/synth_tasks.json")
    ap.add_argument("--db-dir", default="", help="relocate kind=spider2 tasks' SQLite files by schema name (VM paths)")
    ap.add_argument("--dyn-drop", type=int, default=0,
                    help="dynamic sampling: drop a task from the active pool after K consecutive zero-variance groups (0 = off)")
    ap.add_argument("--dyn-readmit", type=int, default=25, help="re-admit a dropped task after this many steps")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tasks-per-step", type=int, default=4)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=2048, help="per-turn cap; matches harness/loop.py NUM_PREDICT")
    ap.add_argument("--rollout", default="hf", choices=["hf", "vllm"],
                    help="hf = serial HF generate (laptop/MPS); vllm = batched vLLM engine on the same GPU (train/vllm_policy.py)")
    ap.add_argument("--vllm-mem", type=float, default=0.35, help="fraction of GPU memory for the vLLM engine")
    ap.add_argument("--vllm-eager", action="store_true", help="skip torch.compile/CUDA graphs: faster start, slower decode")
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false", help="disable activation checkpointing")
    ap.add_argument("--vllm-lora-mode", default="remap", choices=["remap", "textonly"],
                    help="how the HF adapter reaches vLLM's Qwen3.5 (train/vllm_policy.py); pick what train.check_vllm passes")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="disable Qwen thinking (template enable_thinking=False); default on, matching lab.run")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.1, help="success-gated log-length penalty; 0 disables")
    ap.add_argument("--target-chars", type=int, default=6000)
    ap.add_argument("--no-submit-reward", type=float, default=0.0,
                    help="reward for an episode that never calls submit (0 = same as wrong; -1 = SkyRL-SQL)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=25, help="keep a full adapter copy under checkpoints/<tag>/stepN")
    ap.add_argument("--ckpt-every", type=int, default=10, help="resumable checkpoint cadence (adapter + optimizer + RNG)")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<tag>/ckpt/LATEST if present")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--eval-every", type=int, default=25, help="held-out eval cadence in steps; 0 disables")
    ap.add_argument("--eval-tasks", default="", help="JSON task list for eval; default = the 30 hand-made tasks")
    ap.add_argument("--eval-seeds", default="1", help="comma-separated; every seed runs on every eval task, in one batch")
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if a.grad_ckpt:  # recompute activations in backward: long capped trajectories (10 × 2048 tokens) OOM'd without it (gpu5)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    if a.rollout == "hf":
        policy = HFPolicy(model, tok, a.device, max_new_tokens=a.max_new_tokens)
    else:  # after the HF load: initialising vLLM registers its own Qwen3.5 config classes with transformers, and an HF
        # load afterwards fails with "'Qwen3_5Config' object has no attribute 'vocab_size'" (GCP runs gpu2/gpu3, 2026-09-25)
        from train.vllm_policy import VLLMPolicy
        policy = VLLMPolicy(a.model, max_new_tokens=a.max_new_tokens, gpu_memory_utilization=a.vllm_mem,
                            max_lora_rank=max(16, a.lora_r), enforce_eager=a.vllm_eager, lora_mode=a.vllm_lora_mode)

    pool = json.load(open(a.tasks))
    system_tpl = (HARNESS / "system.md").read_text()
    tools = json.loads((HARNESS / "tools.json").read_text())
    Path("runs").mkdir(exist_ok=True)
    log = Path(f"runs/train-{a.tag}.jsonl")
    out_dir = Path(f"checkpoints/{a.tag}")
    eval_tasks = json.load(open(a.eval_tasks)) if a.eval_tasks else fixed_tasks.TASKS
    eval_seeds = [int(s) for s in a.eval_seeds.split(",")]
    for i, task in enumerate(pool):
        task.setdefault("id", f"task{i}")
    pool_state = {"zero_streak": {}, "dropped": {}}  # dynamic sampling state; persisted with each checkpoint
    eval_log = Path(f"runs/train-{a.tag}-eval.jsonl")
    ckpt_root = out_dir / "ckpt"

    def step_dirs(root):  # only stepN dirs; sibling artefacts (e.g. a 'step50-vllm' remap copy) crashed int() on 2026-09-26
        return [q for q in root.glob("step*") if q.is_dir() and re.fullmatch(r"step\d+", q.name)]

    def save_ckpt(step):
        """stepN dir written completely first, LATEST pointer last; keep the newest 3."""
        d = ckpt_root / f"step{step}"
        model.save_pretrained(d)
        torch.save({"step": step, "opt": opt.state_dict(), "py_random": random.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, d / "state.pt")
        (d / "pool_state.json").write_text(json.dumps(pool_state))
        (ckpt_root / "LATEST.tmp").write_text(f"step{step}")
        (ckpt_root / "LATEST.tmp").replace(ckpt_root / "LATEST")
        for old in sorted(step_dirs(ckpt_root), key=lambda q: int(q.name[4:]))[:-3]:
            shutil.rmtree(old, ignore_errors=True)

    def load_ckpt():
        """Newest complete checkpoint (LATEST if valid, else the highest stepN with state.pt), or None."""
        if not ckpt_root.exists():
            return None
        cands = sorted(step_dirs(ckpt_root), key=lambda q: int(q.name[4:]), reverse=True)
        ptr = ckpt_root / "LATEST"
        if ptr.exists():
            cands = [ckpt_root / ptr.read_text().strip()] + cands
        for d in cands:
            if not (d / "adapter_model.safetensors").exists():
                continue
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(model, load_file(str(d / "adapter_model.safetensors")))
            if (d / "state.pt").exists():
                st = torch.load(d / "state.pt", weights_only=False)
                opt.load_state_dict(st["opt"])
                random.setstate(st["py_random"])
                torch.set_rng_state(st["torch_rng"])
                if st.get("cuda_rng") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(st["cuda_rng"])
                return st["step"], d
            # adapter-only (an archived save): weights resume, optimizer moments and RNG restart
            step = int(d.name[4:])
            random.seed(a.seed + step)
            torch.manual_seed(a.seed + step)
            print(json.dumps({"resumed_adapter_only": step, "ckpt": str(d)}), flush=True)
            return step, d
        return None

    def run_eval(step):
        if not a.eval_every:
            return
        t0 = time.time()
        row = {"eval_step": step, **evaluate(policy, tok, system_tpl, tools, eval_tasks, max_turns=a.max_turns, seeds=eval_seeds,
                                             think=a.think, db_dir=a.db_dir),
               "secs": round(time.time() - t0, 1)}
        torch.manual_seed(a.seed + step)  # eval sampling must not perturb the training stream
        model.eval()
        print(json.dumps(row), flush=True)
        with eval_log.open("a") as f:
            f.write(json.dumps(row) + "\n")

    start = 1
    resumed = load_ckpt() if a.resume else None
    if resumed:
        start = resumed[0] + 1
        if a.rollout == "vllm":
            policy.set_adapter(str(resumed[1].resolve()), resumed[0])
        if (resumed[1] / "pool_state.json").exists():
            pool_state = json.loads((resumed[1] / "pool_state.json").read_text())
        print(json.dumps({"resumed_from": resumed[0], "ckpt": str(resumed[1]), "pool_dropped": len(pool_state["dropped"])}), flush=True)
    model.eval()
    if start == 1:
        run_eval(0)
    for step in range(start, a.steps + 1):
        t0 = time.time()
        batch, rewards_all, passes, nosub, turns, groups_kept = [], [], 0, 0, [], 0
        model.eval()
        # dynamic sampling: tasks whose last K groups had one outcome class carry no gradient; rest them for R steps
        for tid, at in list(pool_state["dropped"].items()):
            if step - at >= a.dyn_readmit:
                del pool_state["dropped"][tid]; pool_state["zero_streak"][tid] = 0
        active = [t for t in pool if t["id"] not in pool_state["dropped"]]
        if len(active) < a.tasks_per_step:  # pool exhausted: everything comes back
            pool_state["dropped"].clear(); active = pool
        step_tasks = random.sample(active, a.tasks_per_step)
        envs = [task_env(task, system_tpl, tools, a.db_dir) for task in step_tasks]
        eps = [Episode(tok, system, task["q"], tls, run_sql, max_turns=a.max_turns, think=a.think,
                       seed=a.seed * 1_000_000 + step * 1_000 + k * a.group + g)
               for k, (task, (system, tls, run_sql, _)) in enumerate(zip(step_tasks, envs)) for g in range(a.group)]
        all_trajs = run_episodes(policy, eps)  # every rollout of the step in one lockstep batch
        for k, task in enumerate(step_tasks):
            verify_fn = envs[k][3]
            trajs, rewards = [], []
            for tr in all_trajs[k * a.group:(k + 1) * a.group]:
                res = verify_fn(tr.submitted_sql) if tr.submitted_sql else {"pass": False}
                r = shaped_reward(res["pass"], tr.submitted_sql is not None, tr.gen_chars(), a.target_chars, a.alpha,
                                  a.no_submit_reward)
                trajs.append(tr)
                rewards.append(r)
                passes += int(res["pass"])
                nosub += int(tr.submitted_sql is None)
                turns.append(tr.turns)
            rewards_all.extend(rewards)
            if group_advantages([outcome_class(r) for r in rewards]) is None:  # one outcome class → no signal
                z = pool_state["zero_streak"].get(task["id"], 0) + 1
                pool_state["zero_streak"][task["id"]] = z
                if a.dyn_drop and z >= a.dyn_drop:
                    pool_state["dropped"][task["id"]] = step
                continue
            pool_state["zero_streak"][task["id"]] = 0
            adv = group_advantages(rewards)
            groups_kept += 1
            for tr, av in zip(trajs, adv):
                ids, mask = tokenize_trajectory(tok, tr)
                batch.append((ids, mask, av))
        stats = grpo_step(model, opt, batch, a.device) if batch else {"loss": 0.0, "grad_norm": 0.0}
        if a.rollout == "vllm" and batch:  # hand the updated LoRA to the engine for the next step's rollouts
            live = out_dir / "rollout" / f"step{step}"
            model.save_pretrained(live)
            policy.set_adapter(str(live.resolve()), step)
            for old in (out_dir / "rollout").glob("step*"):  # keep the current and previous hand-off only
                if old.name.split("-")[0] not in (f"step{step}", f"step{step - 1}"):
                    shutil.rmtree(old, ignore_errors=True)
        n = a.tasks_per_step * a.group
        row = {"step": step, "reward_mean": round(sum(rewards_all) / n, 4), "pass_rate": round(passes / n, 4),
               "no_submit_rate": round(nosub / n, 4),
               "mean_turns": round(sum(turns) / n, 2), "groups_kept": groups_kept, "n_traj_in_batch": len(batch),
               "pool_active": len(active) - sum(1 for t in step_tasks if t["id"] in pool_state["dropped"]),
               **{k: round(v, 5) for k, v in stats.items()}, "secs": round(time.time() - t0, 1)}
        print(json.dumps(row), flush=True)
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if step % a.save_every == 0 or step == a.steps:
            model.save_pretrained(out_dir / f"step{step}")
        if a.eval_every and (step % a.eval_every == 0 or step == a.steps):
            run_eval(step)
        if step % a.ckpt_every == 0 or step == a.steps:
            save_ckpt(step)


if __name__ == "__main__":
    main()
