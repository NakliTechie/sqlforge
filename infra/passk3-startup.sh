#!/bin/bash
# Base pass@8 on the climb-2 pool candidates (leg Experiment 11, batch 2): TPC-DS sf0.1 (72 tasks) and BIRD-train candidates
# (lab/birdtrain_cand2.json), 8 seeds each. Base model only. Results → bucket; VM deletes itself.
#   gcloud compute instances create sqlforge-spider2 ... --metadata-from-file startup-script=infra/spider2-eval-startup.sh
# Needs the climb VM gone first (GPUS_ALL_REGIONS quota = 1).
exec > >(tee -a /var/log/passk.log) 2>&1
B=gs://sqlforge-bf3e24-smoke
OUT=$B/passk/climb2-pool3   # fixed per campaign (RUNBOOK §7)
LOG=passk-$(date -u +%Y%m%dT%H%MZ).log
Z=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
NAME=$(hostname)
push(){ gcloud storage cp /var/log/passk.log $OUT/$LOG >/dev/null 2>&1
        [ -d /opt/sq/runs ] && gcloud storage rsync -r /opt/sq/runs $OUT/runs >/dev/null 2>&1; }
finish(){ echo "SPIDER finish $(date -u +%H:%M:%SZ)"; push; gcloud compute instances delete "$NAME" --zone "$Z" --quiet; }
trap finish EXIT
( while true; do sleep 90; push; done ) &

echo "SPIDER boot $(date -u +%H:%M:%SZ) zone=$Z $(nvidia-smi --query-gpu=name --format=csv,noheader)"
mkdir -p /opt/sq && cd /opt/sq && gcloud storage cp $B/climb1/repo.tgz . && tar xzf repo.tgz && mkdir -p runs adapters
gcloud storage rsync -r $OUT/runs runs >/dev/null 2>&1;  echo "SPIDER resume: $(ls runs/*.jsonl 2>/dev/null | wc -l) run logs, $(find runs -name '*.json' -path '*/*/*' | wc -l) episodes already done"
mkdir -p data/tpcds && gcloud storage cp $B/tpcds/tpcds_sf0.1.sqlite data/tpcds/ 2>&1 | grep -i error; echo "TPCDS db $(ls data/tpcds | grep -c sqlite)"
mkdir -p data/bird-train/train/train_databases
for D in $(python3 -c "import json;print(' '.join(sorted({t['schema'] for t in json.load(open('lab/birdtrain_cand2.json'))})))"); do
  mkdir -p data/bird-train/train/train_databases/$D
  gcloud storage cp "$B/bird-train/train_databases/$D/$D.sqlite" data/bird-train/train/train_databases/$D/ 2>&1 | grep -i error
done; echo "BIRDTRAIN dbs $(find data/bird-train -name '*.sqlite' | wc -l)"
export HOME=/root PATH=/root/.local/bin:$PATH
DEBIAN_FRONTEND=noninteractive apt-get install -y -q g++ >/dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync 2>&1 | tail -1; uv pip install vllm ninja "transformers==5.17.0" 2>&1 | tail -1
export PATH=/opt/sq/.venv/bin:$PATH UV_NO_SYNC=1

vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 > /var/log/vllm.log 2>&1 &
for i in $(seq 1 180); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
curl -sf localhost:8000/health >/dev/null || { echo "SPIDER vllm failed"; tail -30 /var/log/vllm.log; exit 0; }
echo "SPIDER vllm healthy $(date -u +%H:%M:%SZ) models: $(curl -s localhost:8000/v1/models | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])')"

run(){ # $1 model name (base or adapter), $2 seeds, $3 tag, $4 taskfile, $5 db-dir
  uv run python -m lab.run --backend openai --model "$1" --taskfile "$4" --db-dir "$5" \
    --seeds "$2" --max-turns 25 --parallel 160 --tag "$3" >> runs/$3.log 2>&1
  echo "SPIDER $3 $(date -u +%H:%M:%SZ) $(grep -E '^METRIC exec_acc' runs/$3.log)"; push
}
echo "SPIDER ready models: $(curl -s localhost:8000/v1/models | python3 -c 'import sys,json;print([m["id"] for m in json.load(sys.stdin)["data"]])') tasks: $(ls lab/tpcds_tasks.json lab/birdtrain_cand2.json | wc -l) birdtrain dbs: $(find data/bird-train -name '*.sqlite' | wc -l)"
[ "$(find data/bird-train -name '*.sqlite' | wc -l)" -gt 0 ] || { echo "SPIDER abort: no BIRD-train databases"; exit 0; }

run Qwen/Qwen3.5-4B 1,2,3,4,5,6,7,8 birdtrain-base lab/birdtrain_cand2.json data/bird-train/train/train_databases
echo "SPIDER done $(date -u +%H:%M:%SZ)" | tee >(gcloud storage cp - $OUT/DONE >/dev/null 2>&1)   # marker: bin/spot-relaunch stops on it
