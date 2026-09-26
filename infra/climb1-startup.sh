#!/bin/bash
# Climb 1 VM job (GCP spot RTX PRO 6000, g4-standard-48). One VM life: restore the newest checkpoint from the
# bucket, train with --resume, sync checkpoints + logs to the bucket every 60 s, write DONE on completion or a
# FAILED marker on a crash, then delete the VM. infra/climb_loop.py (on the Mac) relaunches after preemption.
exec > >(tee -a /var/log/climb.log) 2>&1
B=gs://sqlforge-bf3e24-smoke/climb1
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
  [ -f "$RESTORED_MARK" ] && [ -f /opt/sq/checkpoints/climb1/ckpt/LATEST ] && \
      gcloud storage rsync -r /opt/sq/checkpoints/climb1/ckpt $B/ckpt >/dev/null 2>&1
  for f in /opt/sq/runs/train-climb1.jsonl /opt/sq/runs/train-climb1-eval.jsonl; do
    [ -f "$f" ] && gcloud storage cp "$f" $B/runs/ >/dev/null 2>&1
  done
  [ -f /opt/sq/runs/train-climb1.log ] && gcloud storage cp /opt/sq/runs/train-climb1.log $B/runs/train-climb1-$LIFE.log >/dev/null 2>&1
}
finish() { echo "CLIMB finish $(date -u +%H:%M:%SZ)"; push; gcloud compute instances delete "$NAME" --zone "$Z" --quiet; }
trap finish EXIT
( while true; do sleep 60; push; done ) &

echo "CLIMB boot $(date -u +%H:%M:%SZ) zone=$Z life=$LIFE $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
mkdir -p /opt/sq && cd /opt/sq && gcloud storage cp $B/repo.tgz . && tar xzf repo.tgz && mkdir -p runs checkpoints/climb1/ckpt
export HOME=/root PATH=/root/.local/bin:$PATH
DEBIAN_FRONTEND=noninteractive apt-get install -y -q g++ >/dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync 2>&1 | tail -1
uv pip install vllm ninja flash-linear-attention "transformers==5.17.0" 2>&1 | tail -1
export PATH=/opt/sq/.venv/bin:$PATH UV_NO_SYNC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "CLIMB versions $(uv run python -c 'import torch,transformers,peft,vllm;print("torch",torch.__version__,"transformers",transformers.__version__,"peft",peft.__version__,"vllm",vllm.__version__)' 2>&1 | tail -1)"

# restore: newest checkpoints + the step/eval logs so resumed rows append to one history
gcloud storage rsync -r $B/ckpt checkpoints/climb1/ckpt >/dev/null 2>&1
if [ ! -f checkpoints/climb1/ckpt/LATEST ]; then
  # ckpt/ empty or wiped → fall back to the newest archived adapter (weights only; optimizer/RNG restart)
  S=$(gcloud storage ls $B/saves/ 2>/dev/null | sed -E 's|.*/(step[0-9]+)/$|\1|' | sort -t p -k2 -n | tail -1)
  if [ -n "$S" ]; then
    mkdir -p checkpoints/climb1/ckpt/$S && gcloud storage cp "$B/saves/$S/*" checkpoints/climb1/ckpt/$S/ >/dev/null 2>&1 \
      && echo "$S" > checkpoints/climb1/ckpt/LATEST && echo "CLIMB restore fallback: adapter-only from saves/$S"
  fi
fi
for f in train-climb1.jsonl train-climb1-eval.jsonl; do gcloud storage cp $B/runs/$f runs/ >/dev/null 2>&1; done
touch "$RESTORED_MARK"
echo "CLIMB restored LATEST=$(cat checkpoints/climb1/ckpt/LATEST 2>/dev/null || echo none) $(date -u +%H:%M:%SZ)"
if [ ! -f checkpoints/climb1/ckpt/LATEST ] && [ -s runs/train-climb1.jsonl ]; then
  echo "CLIMB refusing to start from step 0 over an existing history"; tail -3 runs/train-climb1.jsonl | gcloud storage cp - $B/failed/$LIFE-no-ckpt.txt; exit 0
fi

uv run python -m train.run_train --model Qwen/Qwen3.5-4B --tasks lab/climb1_train.json --rollout vllm \
  --vllm-lora-mode remap --vllm-mem 0.45 --steps 200 --tasks-per-step 8 --group 8 --max-turns 25 \
  --device cuda --dtype bf16 --lora-r 32 --lr 1e-5 --no-submit-reward -1 \
  --eval-tasks lab/climb1_eval.json --eval-every 25 --ckpt-every 10 --save-every 1000 --tag climb1 --resume \
  > runs/train-climb1.log 2>&1
RC=$?
echo "CLIMB trainer exit $RC $(date -u +%H:%M:%SZ) LATEST=$(cat checkpoints/climb1/ckpt/LATEST 2>/dev/null)"
push
if [ $RC -eq 0 ] && [ "$(cat checkpoints/climb1/ckpt/LATEST 2>/dev/null)" = "step200" ]; then
  echo "step200 $(date -u +%FT%TZ)" | gcloud storage cp - $B/DONE
else
  tail -40 runs/train-climb1.log | gcloud storage cp - $B/failed/$LIFE-rc$RC.txt
fi
