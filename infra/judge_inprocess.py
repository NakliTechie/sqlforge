"""Judge diagnostic (cold review 2026-09-27, Opus C1): run the TRAINER's own rollout path — train.run_train.evaluate over an
in-process vLLM engine, prior thinking kept in context, top_p 0.95, merged tool messages — on a benchmark taskfile, for the
base model and for one adapter, with per-task outcomes. lab.run (the primary judge) drops prior thinking and sends one tool
message per call; if lab.run shows a null and this path shows movement, the gap is a context-format shift, not "no transfer".

  uv run python infra/judge_inprocess.py --tasks lab/spider2_sqlite.json --db-dir data/spider2-lite/localdb \\
      --seeds 1,2,3 --adapter adapters/step150 --tag spider2-inprocess
Writes runs/<tag>-base.json, runs/<tag>-adapter.json (summaries) and runs/<tag>-tasks.jsonl (per task × seed × model).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a script from infra/: the repo root must be importable
# (2026-09-28 00:08: the judge's diagnostic died in 24 s with "No module named 'train'")

HARNESS = Path(__file__).resolve().parent.parent / "harness"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--db-dir", default="")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--adapter", default="", help="PEFT adapter dir (raw save); empty = base only")
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--vllm-mem", type=float, default=0.85)
    ap.add_argument("--tag", default="inprocess")
    a = ap.parse_args()
    from transformers import AutoTokenizer  # tokenizer before the engine (config-registration order, see run_train.py)
    tok = AutoTokenizer.from_pretrained(a.model)
    from train.run_train import evaluate
    from train.vllm_policy import VLLMPolicy
    policy = VLLMPolicy(a.model, max_new_tokens=2048, gpu_memory_utilization=a.vllm_mem, max_lora_rank=32, lora_mode="remap")
    tasks = json.load(open(a.tasks))
    for t in tasks:
        t.setdefault("set", "all")
    system_tpl = (HARNESS / "system.md").read_text()
    tools = json.loads((HARNESS / "tools.json").read_text())
    seeds = [int(s) for s in a.seeds.split(",")]
    Path("runs").mkdir(exist_ok=True)
    per_task = Path(f"runs/{a.tag}-tasks.jsonl")
    arms = [("base", None)] + ([("adapter", a.adapter)] if a.adapter else [])
    for name, adapter in arms:
        if adapter:
            policy.set_adapter(str(Path(adapter).resolve()), 150)
        t0 = time.time()
        # tag per-task rows with the arm via eval_step: 0 = base, 1 = adapter
        summary = evaluate(policy, tok, system_tpl, tools, tasks, max_turns=a.max_turns, seeds=seeds, think=True,
                           db_dir=a.db_dir, per_task_log=per_task, eval_step=0 if name == "base" else 1)
        summary["arm"], summary["adapter"], summary["secs"] = name, adapter, round(time.time() - t0, 1)
        json.dump(summary, open(f"runs/{a.tag}-{name}.json", "w"), indent=1)
        print(f"METRIC {a.tag} {name} exec_acc {summary['exec_acc']} by_seed {summary['exec_acc_by_seed']} nosub "
              f"{summary['no_submit_rate']} turns {summary['mean_turns']} secs {summary['secs']}", flush=True)


if __name__ == "__main__":
    main()
