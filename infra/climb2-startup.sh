#!/bin/bash
# Climb 2 VM job (from climb1-startup.sh; pool = lab/climb2_pool.json on real SQLite schemas, steering eval = BIRD challenging × 3 seeds) (GCP spot RTX PRO 6000, g4-standard-48). One VM life: restore the newest checkpoint from the
# bucket, train with --resume, sync checkpoints + logs to the bucket every 60 s, write DONE on completion or a
# FAILED marker on a crash, then delete the VM. infra/climb_loop.py (on the Mac) relaunches after preemption.
exec > >(tee -a /var/log/climb.log) 2>&1
ROOT=gs://sqlforge-bf3e24-smoke   # gcloud storage does not resolve ".." in gs:// paths: lives 1–2 (2026-09-27 01:14) synced nothing and aborted
B=$ROOT/climb2
Z=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
NAME=$(hostname)
LIFE=$(date -u +%Y%m%dT%H%M%SZ)
cd /opt 2>/dev/null || mkdir -p /opt
# restore-done marker is a FILE: the push loop below is a forked subshell, so a shell variable set after the fork is
# never seen by it (2026-09-26: a RESTORED variable was invisible to the loop; no checkpoint synced during VM life 5)
RESTORED_MARK=/opt/sq/.restored
push() {
  gcloud storage cp /var/log/climb.log $B/vm-logs/$LIFE-$NAME.log >/dev/null 2>&1
  # ckpt sync only after restore, and never deleting bucket objects: on 2026-09-26 an early push with an empty local
  # ckpt dir + --delete-unmatched-destination-objects wiped the bucket's ckpt/ (LATEST, step40-60) — 5 lives lost
  [ -f "$RESTORED_MARK" ] && [ -f /opt/sq/checkpoints/climb2/ckpt/LATEST ] && \
      gcloud storage rsync -r /opt/sq/checkpoints/climb2/ckpt $B/ckpt >/dev/null 2>&1
  for f in /opt/sq/runs/train-climb2.jsonl /opt/sq/runs/train-climb2-eval.jsonl /opt/sq/runs/train-climb2-eval-tasks.jsonl; do
    [ -f "$f" ] && gcloud storage cp "$f" $B/runs/ >/dev/null 2>&1
  done
  [ -f /opt/sq/runs/train-climb2.log ] && gcloud storage cp /opt/sq/runs/train-climb2.log $B/runs/train-climb2-$LIFE.log >/dev/null 2>&1
}
finish() { echo "CLIMB finish $(date -u +%H:%M:%SZ)"; push; gcloud compute instances delete "$NAME" --zone "$Z" --quiet; }
trap finish EXIT
( while true; do sleep 60; push; done ) &

echo "CLIMB boot $(date -u +%H:%M:%SZ) zone=$Z life=$LIFE $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
mkdir -p /opt/sq && cd /opt/sq && gcloud storage cp $B/repo.tgz . && tar xzf repo.tgz && mkdir -p runs checkpoints/climb2/ckpt
# data: the pool's SQLite files (TPC-H, TPC-DS, the BIRD-train databases the pool uses) + BIRD Mini-Dev for the steering eval
mkdir -p data/tpch data/tpcds data/bird-train/train/train_databases data/bird-minidev/dev_databases
gcloud storage cp $ROOT/tpch/tpch_sf0.1.sqlite data/tpch/ 2>&1 | grep -i error; gcloud storage cp $ROOT/tpcds/tpcds_sf0.1.sqlite data/tpcds/ 2>&1 | grep -i error
for D in $(python3 -c "import json;print(' '.join(sorted({t['schema'] for t in json.load(open('lab/climb2_pool.json')) if t['set']=='birdtrain'})))"); do
  mkdir -p data/bird-train/train/train_databases/$D
  gcloud storage cp "$ROOT/bird-train/train_databases/$D/$D.sqlite" data/bird-train/train/train_databases/$D/ 2>&1 | grep -i error
done
gcloud storage rsync -r $ROOT/bird/dev_databases data/bird-minidev/dev_databases >/dev/null 2>&1
echo "CLIMB data: tpch $(ls data/tpch | wc -l) tpcds $(ls data/tpcds | wc -l) birdtrain $(find data/bird-train -name '*.sqlite' | wc -l) minidev $(find data/bird-minidev -name '*.sqlite' | wc -l)"
[ "$(find data/bird-train -name '*.sqlite' | wc -l)" -gt 40 ] || { echo "CLIMB abort: pool databases missing"; exit 0; }
export HOME=/root PATH=/root/.local/bin:$PATH
DEBIAN_FRONTEND=noninteractive apt-get install -y -q g++ >/dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync 2>&1 | tail -1
uv pip install vllm ninja flash-linear-attention "transformers==5.17.0" 2>&1 | tail -1
export PATH=/opt/sq/.venv/bin:$PATH UV_NO_SYNC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "CLIMB versions $(uv run python -c 'import torch,transformers,peft,vllm;print("torch",torch.__version__,"transformers",transformers.__version__,"peft",peft.__version__,"vllm",vllm.__version__)' 2>&1 | tail -1)"

# restore: newest checkpoints + the step/eval logs so resumed rows append to one history
gcloud storage rsync -r $B/ckpt checkpoints/climb2/ckpt >/dev/null 2>&1
if [ ! -f checkpoints/climb2/ckpt/LATEST ]; then
  # ckpt/ empty or wiped → fall back to the newest archived adapter (weights only; optimizer/RNG restart)
  S=$(gcloud storage ls $B/saves/ 2>/dev/null | sed -E 's|.*/(step[0-9]+)/$|\1|' | sort -t p -k2 -n | tail -1)
  if [ -n "$S" ]; then
    mkdir -p checkpoints/climb2/ckpt/$S && gcloud storage cp "$B/saves/$S/*" checkpoints/climb2/ckpt/$S/ >/dev/null 2>&1 \
      && echo "$S" > checkpoints/climb2/ckpt/LATEST && echo "CLIMB restore fallback: adapter-only from saves/$S"
  fi
fi
for f in train-climb2.jsonl train-climb2-eval.jsonl train-climb2-eval-tasks.jsonl; do gcloud storage cp $B/runs/$f runs/ >/dev/null 2>&1; done
touch "$RESTORED_MARK"
echo "CLIMB restored LATEST=$(cat checkpoints/climb2/ckpt/LATEST 2>/dev/null || echo none) $(date -u +%H:%M:%SZ)"
if [ ! -f checkpoints/climb2/ckpt/LATEST ] && [ -s runs/train-climb2.jsonl ]; then
  echo "CLIMB refusing to start from step 0 over an existing history"; tail -3 runs/train-climb2.jsonl | gcloud storage cp - $B/failed/$LIFE-no-ckpt.txt; exit 0
fi

uv run python -m train.run_train --model Qwen/Qwen3.5-4B --tasks lab/climb2_pool.json --db-dir data --rollout vllm \
  --vllm-lora-mode remap --vllm-mem 0.40 --steps 150 --tasks-per-step 8 --group 8 --max-turns 25 \
  --device cuda --dtype bf16 --lora-r 32 --lr 1e-5 --no-submit-reward -1 --dyn-drop 2 --dyn-readmit 25 \
  --eval-tasks lab/bird_challenging.json --eval-seeds 1,2,3 --eval-every 25 --ckpt-every 10 --save-every 1000 --tag climb2 --resume \
  > runs/train-climb2.log 2>&1
RC=$?
echo "CLIMB trainer exit $RC $(date -u +%H:%M:%SZ) LATEST=$(cat checkpoints/climb2/ckpt/LATEST 2>/dev/null)"
push
if [ $RC -eq 0 ] && [ "$(cat checkpoints/climb2/ckpt/LATEST 2>/dev/null)" = "step150" ]; then
  echo "step150 $(date -u +%FT%TZ)" | gcloud storage cp - $B/DONE
else
  tail -40 runs/train-climb2.log | gcloud storage cp - $B/failed/$LIFE-rc$RC.txt
fi
