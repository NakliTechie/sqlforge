#!/bin/bash
# Spider 2.0-Lite SQLite eval on GCP (spot RTX PRO 6000): base Qwen3.5-4B + every archived climb-2 adapter, via one
# vLLM server with LoRA modules, lab.run --backend openai, all 135 tasks concurrent. Results → bucket; VM deletes itself.
#   gcloud compute instances create sqlforge-spider2 ... --metadata-from-file startup-script=infra/spider2-eval-startup.sh
# Needs the climb VM gone first (GPUS_ALL_REGIONS quota = 1).
exec > >(tee -a /var/log/spider2.log) 2>&1
B=gs://sqlforge-bf3e24-smoke
OUT=$B/spider2/eval-climb2   # fixed per campaign: a preempted life resumes from the runs/ already pushed here
LOG=spider2-$(date -u +%Y%m%dT%H%MZ).log
Z=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
NAME=$(hostname)
push(){ gcloud storage cp /var/log/spider2.log $OUT/$LOG >/dev/null 2>&1
        [ -d /opt/sq/runs ] && gcloud storage rsync -r /opt/sq/runs $OUT/runs >/dev/null 2>&1; }
finish(){ echo "SPIDER finish $(date -u +%H:%M:%SZ)"; push; gcloud compute instances delete "$NAME" --zone "$Z" --quiet; }
trap finish EXIT
( while true; do sleep 90; push; done ) &

echo "SPIDER boot $(date -u +%H:%M:%SZ) zone=$Z $(nvidia-smi --query-gpu=name --format=csv,noheader)"
mkdir -p /opt/sq && cd /opt/sq && gcloud storage cp $B/climb2/repo.tgz . && tar xzf repo.tgz && mkdir -p runs data/spider2-lite/localdb adapters
gcloud storage rsync -r $OUT/runs runs >/dev/null 2>&1; echo "SPIDER resume: $(ls runs/*.jsonl 2>/dev/null | wc -l) run logs, $(find runs -name '*.json' -path '*/*/*' | wc -l) episodes already done"
gcloud storage rsync -r $B/spider2/localdb data/spider2-lite/localdb >/dev/null 2>&1; echo "SPIDER dbs $(ls data/spider2-lite/localdb | wc -l)"
mkdir -p data/bird-minidev/dev_databases && gcloud storage rsync -r $B/bird/dev_databases data/bird-minidev/dev_databases >/dev/null 2>&1; echo "BIRD dbs $(find data/bird-minidev -name '*.sqlite' | wc -l)"
export HOME=/root PATH=/root/.local/bin:$PATH
DEBIAN_FRONTEND=noninteractive apt-get install -y -q g++ >/dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync 2>&1 | tail -1; uv pip install vllm ninja "transformers==5.17.0" 2>&1 | tail -1
export PATH=/opt/sq/.venv/bin:$PATH UV_NO_SYNC=1

# adapters: every archived save, remapped to vLLM's layout (train.vllm_policy.export_adapter)
LORA_ARGS=""
for S in $(gcloud storage ls $B/climb2/saves/ | sed -E 's|.*/(step[0-9]+)/$|\1|' | sort -t p -k2 -n); do
  mkdir -p adapters/$S  # gcloud storage cp refuses a destination dir that does not exist
  gcloud storage cp "$B/climb2/saves/$S/*" adapters/$S/ 2>&1 | grep -i error
  if uv run python -c "from train.vllm_policy import export_adapter; export_adapter('adapters/$S', 'adapters/$S-vllm')"; then
    LORA_ARGS="$LORA_ARGS $S=/opt/sq/adapters/$S-vllm"
  else echo "SPIDER adapter $S FAILED (ls: $(ls adapters/$S 2>&1 | tr '\n' ' '))"; fi
done
echo "SPIDER adapters:$LORA_ARGS"
[ -n "$LORA_ARGS" ] || { echo "SPIDER no adapters exported; aborting (base-only run is not what this job is for)"; exit 0; }
vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.85 --enable-lora --max-lora-rank 32 \
  --max-loras 2 --lora-modules $LORA_ARGS --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 > /var/log/vllm.log 2>&1 &
for i in $(seq 1 180); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
curl -sf localhost:8000/health >/dev/null || { echo "SPIDER vllm failed"; tail -30 /var/log/vllm.log; exit 0; }
echo "SPIDER vllm healthy $(date -u +%H:%M:%SZ) models: $(curl -s localhost:8000/v1/models | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])')"

run(){ # $1 model name (base or adapter), $2 seeds, $3 tag, $4 taskfile, $5 db-dir
  uv run python -m lab.run --backend openai --model "$1" --taskfile "$4" --db-dir "$5" \
    --seeds "$2" --max-turns 25 --parallel 160 --tag "$3" >> runs/$3.log 2>&1
  echo "SPIDER $3 $(date -u +%H:%M:%SZ) $(grep -E '^METRIC exec_acc' runs/$3.log)"; push
}
ADAPTERS=$(echo $LORA_ARGS | tr ' ' '\n' | cut -d= -f1); LAST=$(echo "$ADAPTERS" | tail -1)
for BENCH in "spider2 lab/spider2_sqlite.json data/spider2-lite/localdb" "bird lab/bird_minidev.json data/bird-minidev/dev_databases"; do
  set -- $BENCH; BN=$1; TF=$2; DBD=$3
  [ -f "$TF" ] || { echo "SPIDER skip $BN: no $TF"; continue; }
  run Qwen/Qwen3.5-4B 1,2,3 "$BN-base" "$TF" "$DBD"
  for S in $ADAPTERS; do run "$S" 1 "$BN-$S" "$TF" "$DBD"; done
  [ -n "$LAST" ] && run "$LAST" 2,3 "$BN-$LAST-s23" "$TF" "$DBD"
done
echo "SPIDER done $(date -u +%H:%M:%SZ)" | tee >(gcloud storage cp - $OUT/DONE >/dev/null 2>&1)   # marker: bin/spot-relaunch stops on it
