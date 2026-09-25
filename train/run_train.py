"""One GRPO climb over the synthesised task pool.

uv run python -m train.run_train --model Qwen/Qwen3.5-0.8B --tasks lab/synth_tasks.json \
    --steps 1 --tasks-per-step 1 --group 2 --max-turns 6 --device mps --tag smoke

Every step: sample B tasks → G rollouts each with the current LoRA policy → verify (hidden snapshots)
→ shaped reward → group advantages → one optimizer step. Logs a JSON line per step to runs/train-<tag>.jsonl;
saves the adapter every --save-every steps.

Held-out eval: at step 0 (baseline), every --eval-every steps and at the last step, the current policy runs
the 30 hand-made tasks once each (seed --eval-seed) and the exec_acc / by-hops / cap-rate line goes to
runs/train-<tag>-eval.jsonl. --eval-every 0 disables.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab import schemas, tasks as fixed_tasks, verify
from train.grpo import group_advantages, grpo_step, shaped_reward
from train.rollout import HFPolicy, run_episode, tokenize_trajectory

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


def evaluate(policy, tok, system_tpl, tools, eval_tasks, *, max_turns: int, seed: int) -> dict:
    """One rollout per held-out task with the current policy; returns the runner-style summary."""
    torch.manual_seed(seed)
    policy.model.eval()
    rows = []
    for task in eval_tasks:
        system = system_tpl.replace("{ddl}", task["ddl"].strip())
        tr = run_episode(policy, tok, system, task["q"], tools, make_run_sql(task["schema"]), max_turns=max_turns)
        res = verify.verify(task, tr.submitted_sql, fixed_tasks.HIDDEN_SEEDS) if tr.submitted_sql else {"pass": False}
        rows.append({"hops": task["hops"], "pass": bool(res["pass"]), "hit_cap": tr.hit_cap, "turns": tr.turns})
    n = len(rows)
    hops = sorted({r["hops"] for r in rows})
    return {"exec_acc": round(sum(r["pass"] for r in rows) / n, 4),
            "exec_acc_by_hops": {h: round(sum(r["pass"] for r in rows if r["hops"] == h) / sum(1 for r in rows if r["hops"] == h), 3) for h in hops},
            "turn_cap_rate": round(sum(r["hit_cap"] for r in rows) / n, 4),
            "mean_turns": round(sum(r["turns"] for r in rows) / n, 2), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tasks", default="lab/synth_tasks.json")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tasks-per-step", type=int, default=4)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.1, help="success-gated log-length penalty; 0 disables")
    ap.add_argument("--target-chars", type=int, default=6000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--eval-every", type=int, default=25, help="held-out eval cadence in steps; 0 disables")
    ap.add_argument("--eval-tasks", default="", help="JSON task list for eval; default = the 30 hand-made tasks")
    ap.add_argument("--eval-seed", type=int, default=1)
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(a.device)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    policy = HFPolicy(model, tok, a.device, max_new_tokens=a.max_new_tokens)

    pool = json.load(open(a.tasks))
    system_tpl = (HARNESS / "system.md").read_text()
    tools = json.loads((HARNESS / "tools.json").read_text())
    Path("runs").mkdir(exist_ok=True)
    log = Path(f"runs/train-{a.tag}.jsonl")
    out_dir = Path(f"checkpoints/{a.tag}")
    eval_tasks = json.load(open(a.eval_tasks)) if a.eval_tasks else fixed_tasks.TASKS
    eval_log = Path(f"runs/train-{a.tag}-eval.jsonl")

    def run_eval(step):
        if not a.eval_every:
            return
        t0 = time.time()
        row = {"eval_step": step, **evaluate(policy, tok, system_tpl, tools, eval_tasks, max_turns=a.max_turns, seed=a.eval_seed),
               "secs": round(time.time() - t0, 1)}
        torch.manual_seed(a.seed + step)  # eval sampling must not perturb the training stream
        print(json.dumps(row), flush=True)
        with eval_log.open("a") as f:
            f.write(json.dumps(row) + "\n")

    run_eval(0)
    for step in range(1, a.steps + 1):
        t0 = time.time()
        batch, rewards_all, passes, turns, groups_kept = [], [], 0, [], 0
        for task in random.sample(pool, a.tasks_per_step):
            system = system_tpl.replace("{ddl}", task["ddl"].strip())
            trajs, rewards = [], []
            model.eval()
            for g in range(a.group):
                tr = run_episode(policy, tok, system, task["q"], tools, make_run_sql(task["schema"]), max_turns=a.max_turns)
                res = verify.verify(task, tr.submitted_sql, fixed_tasks.HIDDEN_SEEDS) if tr.submitted_sql else {"pass": False}
                r = shaped_reward(res["pass"], tr.gen_chars(), a.target_chars, a.alpha)
                trajs.append(tr)
                rewards.append(r)
                passes += int(res["pass"])
                turns.append(tr.turns)
            rewards_all.extend(rewards)
            adv = group_advantages([1.0 if r > 0 else 0.0 for r in rewards])  # zero-variance filter on raw pass/fail
            if adv is None:
                continue
            adv = group_advantages(rewards)
            groups_kept += 1
            for tr, av in zip(trajs, adv):
                ids, mask = tokenize_trajectory(tok, tr)
                batch.append((ids, mask, av))
        stats = grpo_step(model, opt, batch, a.device) if batch else {"loss": 0.0, "grad_norm": 0.0}
        n = a.tasks_per_step * a.group
        row = {"step": step, "reward_mean": round(sum(rewards_all) / n, 4), "pass_rate": round(passes / n, 4),
               "mean_turns": round(sum(turns) / n, 2), "groups_kept": groups_kept, "n_traj_in_batch": len(batch),
               **{k: round(v, 5) for k, v in stats.items()}, "secs": round(time.time() - t0, 1)}
        print(json.dumps(row), flush=True)
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if step % a.save_every == 0 or step == a.steps:
            model.save_pretrained(out_dir / f"step{step}")
        if a.eval_every and (step % a.eval_every == 0 or step == a.steps):
            run_eval(step)


if __name__ == "__main__":
    main()
