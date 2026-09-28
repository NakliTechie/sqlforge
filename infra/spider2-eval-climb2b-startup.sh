#!/bin/bash
# Climb-2 judge, second life (2026-09-28, Chirag: option A): (1) Spider 2.0-Lite SQLite, adapters step60/80/100 × seeds
# 2-5 (seed 1 and base × 8 are in spider2/eval-climb2) — is the single-seed step-80 gain (+4.5 [+0.5, +8.2]) real?
# (2) the in-process diagnostic (cold review C1): trainer's own rollout path, base + step150 × 3 seeds on the 135.
# Results → bucket; VM deletes itself. Launched under bin/spot-relaunch (sqlforge-spider2-eval2b.env).
exec > >(tee -a /var/log/spider2.log) 2>&1
B=gs://sqlforge-bf3e24-smoke
OUT=$B/spider2/eval-climb2b   # fixed per campaign: a preempted life resumes from the runs/ already pushed here
LOG=spider2-$(date -u +%Y%m%dT%H%MZ).log
Z=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
readonly SELF=$(hostname)
push(){ gcloud storage cp /var/log/spider2.log $OUT/$LOG >/dev/null 2>&1
        [ -d /opt/sq/runs ] && gcloud storage rsync -r /opt/sq/runs $OUT/runs >/dev/null 2>&1; }
finish(){ echo "SPIDER finish $(date -u +%H:%M:%SZ)"; push; gcloud compute instances delete "$SELF" --zone "$Z" --quiet; }
trap finish EXIT
( while true; do sleep 90; push; done ) &

echo "SPIDER boot $(date -u +%H:%M:%SZ) zone=$Z $(nvidia-smi --query-gpu=name --format=csv,noheader)"
mkdir -p /opt/sq && cd /opt/sq && gcloud storage cp $OUT/repo.tgz . && tar xzf repo.tgz && mkdir -p runs data/spider2-lite/localdb adapters
gcloud storage rsync -r $OUT/runs runs >/dev/null 2>&1; echo "SPIDER resume: $(ls runs/*.jsonl 2>/dev/null | wc -l) run logs, $(find runs -name '*.json' -path '*/*/*' | wc -l) episodes already done"
gcloud storage rsync -r $B/spider2/localdb data/spider2-lite/localdb >/dev/null 2>&1; echo "SPIDER dbs $(ls data/spider2-lite/localdb | wc -l)"
export HOME=/root PATH=/root/.local/bin:$PATH
DEBIAN_FRONTEND=noninteractive apt-get install -y -q g++ >/dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync 2>&1 | tail -1; uv pip install "vllm==0.30.0" ninja "transformers==5.17.0" 2>&1 | tail -1
export PATH=/opt/sq/.venv/bin:$PATH UV_NO_SYNC=1

LORA_ARGS=""
for S in step60 step80 step100 step150; do
  mkdir -p adapters/$S
  gcloud storage cp "$B/climb2/saves/$S/*" adapters/$S/ 2>&1 | grep -i error
  if uv run python -c "from train.vllm_policy import export_adapter; export_adapter('adapters/$S', 'adapters/$S-vllm')"; then
    LORA_ARGS="$LORA_ARGS $S=/opt/sq/adapters/$S-vllm"
  else echo "SPIDER adapter $S FAILED (ls: $(ls adapters/$S 2>&1 | tr '\n' ' '))"; fi
done
echo "SPIDER adapters:$LORA_ARGS"
[ -n "$LORA_ARGS" ] || { echo "SPIDER no adapters exported; aborting"; exit 0; }
vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.85 --enable-lora --max-lora-rank 32 \
  --max-loras 2 --lora-modules $LORA_ARGS --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 > /var/log/vllm.log 2>&1 &
for i in $(seq 1 180); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
curl -sf localhost:8000/health >/dev/null || { echo "SPIDER vllm failed"; tail -30 /var/log/vllm.log; exit 0; }
echo "SPIDER vllm healthy $(date -u +%H:%M:%SZ) models: $(curl -s localhost:8000/v1/models | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])')"

run(){ # $1 model name, $2 seeds, $3 tag
  uv run python -m lab.run --backend openai --model "$1" --taskfile lab/spider2_sqlite.json --db-dir data/spider2-lite/localdb \
    --seeds "$2" --max-turns 25 --parallel 160 --tag "$3" >> runs/$3.log 2>&1
  echo "SPIDER $3 $(date -u +%H:%M:%SZ) $(grep -E '^METRIC exec_acc' runs/$3.log)"; push
}
for S in step60 step80 step100; do
  echo "$LORA_ARGS" | grep -q "$S=" && run "$S" 2,3,4,5 "spider2-$S-s2345"
done
# diagnostic (cold review C1): the trainer's rollout path (prior thinking kept in context, top_p 0.95, merged tool messages),
# base and step150, 3 seeds each — separates "no transfer" from "context-format shift"
pkill -f "vllm serve" ; sleep 15
if [ -f runs/spider2-inprocess-adapter.json ]; then echo "SPIDER inprocess already done (resume)"; else
  uv run python infra/judge_inprocess.py --tasks lab/spider2_sqlite.json --db-dir data/spider2-lite/localdb --seeds 1,2,3 \
    --adapter adapters/step150 --tag spider2-inprocess > runs/spider2-inprocess.log 2>&1
  echo "SPIDER inprocess rc=$? $(date -u +%H:%M:%SZ) $(grep -E '^METRIC' runs/spider2-inprocess.log | tr '\n' ' ')"; tail -3 runs/spider2-inprocess.log; push
fi
echo "SPIDER done $(date -u +%H:%M:%SZ)" | tee >(gcloud storage cp - $OUT/DONE >/dev/null 2>&1)
