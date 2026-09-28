#!/bin/bash
# Ablation 1 (2026-09-28, Chirag: "do a, then c, then stop"): climb-2 recipe with the commitment reward removed
# (--no-submit-reward 0; a group of only {wrong, no-submit} is then zero-variance and dropped), 40 GRPO steps on the climb-2
# pool, no steering evals; then the in-harness judge (infra/judge_inprocess.py, adapter arm only — the base arm for Spider
# exists in spider2/eval-climb2b). One VM life does whatever remains: resume the checkpoint, train to step 40, judge, DONE.
# Under bin/spot-relaunch (sqlforge-ablate1.env). Pre-registration: plan/lab/task-synth/2026-09-24-leg.md Experiment 13.
exec > >(tee -a /var/log/ablate.log) 2>&1
ROOT=gs://sqlforge-bf3e24-smoke
B=$ROOT/ablate1
Z=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
readonly SELF=$(hostname)
LIFE=$(date -u +%Y%m%dT%H%M%SZ)
RESTORED_MARK=/opt/sq/.restored
push() {
  gcloud storage cp /var/log/ablate.log $B/vm-logs/$LIFE-$SELF.log >/dev/null 2>&1
  [ -f "$RESTORED_MARK" ] && [ -f /opt/sq/checkpoints/ablate1/ckpt/LATEST ] && \
      gcloud storage rsync -r /opt/sq/checkpoints/ablate1/ckpt $B/ckpt >/dev/null 2>&1
  [ -d /opt/sq/runs ] && gcloud storage rsync -r /opt/sq/runs $B/runs >/dev/null 2>&1
}
finish() { echo "ABLATE finish $(date -u +%H:%M:%SZ)"; push; gcloud compute instances delete "$SELF" --zone "$Z" --quiet; }
trap finish EXIT
( while true; do sleep 60; push; done ) &

echo "ABLATE boot $(date -u +%H:%M:%SZ) zone=$Z life=$LIFE $(nvidia-smi --query-gpu=name --format=csv,noheader)"
mkdir -p /opt/sq && cd /opt/sq && gcloud storage cp $B/repo.tgz . && tar xzf repo.tgz && mkdir -p runs checkpoints/ablate1/ckpt
mkdir -p data/tpch data/tpcds data/bird-train/train/train_databases data/spider2-lite/localdb
gcloud storage cp $ROOT/tpch/tpch_sf0.1.sqlite data/tpch/ 2>&1 | grep -i error; gcloud storage cp $ROOT/tpcds/tpcds_sf0.1.sqlite data/tpcds/ 2>&1 | grep -i error
for D in $(python3 -c "import json;print(' '.join(sorted({t['schema'] for t in json.load(open('lab/climb2_pool.json')) if t['set']=='birdtrain'})))"); do
  mkdir -p data/bird-train/train/train_databases/$D
  gcloud storage cp "$ROOT/bird-train/train_databases/$D/$D.sqlite" data/bird-train/train/train_databases/$D/ 2>&1 | grep -i error
done
gcloud storage rsync -r $ROOT/spider2/localdb data/spider2-lite/localdb >/dev/null 2>&1
echo "ABLATE data: tpch $(ls data/tpch | wc -l) tpcds $(ls data/tpcds | wc -l) birdtrain $(find data/bird-train -name '*.sqlite' | wc -l) spider $(ls data/spider2-lite/localdb | wc -l)"
[ "$(find data/bird-train -name '*.sqlite' | wc -l)" -gt 40 ] || { echo "ABLATE abort: pool databases missing"; exit 0; }
export HOME=/root PATH=/root/.local/bin:$PATH
DEBIAN_FRONTEND=noninteractive apt-get install -y -q g++ >/dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync 2>&1 | tail -1
uv pip install "vllm==0.30.0" ninja flash-linear-attention "transformers==5.17.0" 2>&1 | tail -1
export PATH=/opt/sq/.venv/bin:$PATH UV_NO_SYNC=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

gcloud storage rsync -r $B/ckpt checkpoints/ablate1/ckpt >/dev/null 2>&1
gcloud storage rsync -r $B/runs runs >/dev/null 2>&1
touch "$RESTORED_MARK"
echo "ABLATE restored LATEST=$(cat checkpoints/ablate1/ckpt/LATEST 2>/dev/null || echo none) $(date -u +%H:%M:%SZ)"
if [ ! -f checkpoints/ablate1/ckpt/LATEST ] && [ -s runs/train-ablate1.jsonl ]; then
  echo "ABLATE refusing to start from step 0 over an existing history"; exit 0
fi

if [ "$(cat checkpoints/ablate1/ckpt/LATEST 2>/dev/null)" != "step40" ]; then
  uv run python -m train.run_train --model Qwen/Qwen3.5-4B --tasks lab/climb2_pool.json --db-dir data --rollout vllm \
    --vllm-lora-mode remap --vllm-mem 0.40 --steps 40 --tasks-per-step 8 --group 8 --max-turns 25 \
    --device cuda --dtype bf16 --lora-r 32 --lr 1e-5 --no-submit-reward 0 --dyn-drop 2 --dyn-readmit 25 \
    --eval-every 0 --ckpt-every 5 --save-every 1000 --tag ablate1 --resume >> runs/train-ablate1.log 2>&1
  RC=$?; echo "ABLATE trainer exit $RC $(date -u +%H:%M:%SZ) LATEST=$(cat checkpoints/ablate1/ckpt/LATEST 2>/dev/null)"; push
  [ "$(cat checkpoints/ablate1/ckpt/LATEST 2>/dev/null)" = "step40" ] || { tail -40 runs/train-ablate1.log | gcloud storage cp - $B/failed/$LIFE-rc$RC.txt; exit 0; }
fi
sleep 20   # let the trainer's engine release the GPU
if [ ! -f runs/spider2-inprocess-ablate1-adapter.json ]; then
  uv run python infra/judge_inprocess.py --tasks lab/spider2_sqlite.json --db-dir data/spider2-lite/localdb --seeds 1,2,3 \
    --adapter checkpoints/ablate1/ckpt/step40 --arms adapter --tag spider2-inprocess-ablate1 > runs/spider2-inprocess-ablate1.log 2>&1
  echo "ABLATE judge rc=$? $(date -u +%H:%M:%SZ) $(grep -E '^METRIC' runs/spider2-inprocess-ablate1.log | tr '\n' ' ')"; tail -3 runs/spider2-inprocess-ablate1.log; push
fi
[ -f runs/spider2-inprocess-ablate1-adapter.json ] && echo "ABLATE done $(date -u +%H:%M:%SZ)" | tee >(gcloud storage cp - $B/DONE >/dev/null 2>&1)
