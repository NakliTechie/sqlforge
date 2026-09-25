"""GCP smoke probe against a local vLLM OpenAI server (run on the VM by infra/gcp-smoke.yaml).

1. One harness-shaped chat (system.md + a hand-made task + tools.json, thinking on): does vLLM return reasoning
   separately and a structured tool call?
2. Throughput: 32 concurrent single-turn chats, 512 max tokens each → aggregate completion tokens/s.
Prints one JSON line per probe prefixed SMOKE so the log is greppable.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab import tasks  # noqa: E402

task = tasks.TASKS[0]
system = Path("harness/system.md").read_text().replace("{ddl}", task["ddl"].strip())
tools = json.loads(Path("harness/tools.json").read_text())
base = {"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": task["q"]}],
        "tools": tools, "temperature": 0.6, "chat_template_kwargs": {"enable_thinking": True}}

t0 = time.time()
r = requests.post(URL, json={**base, "max_tokens": 2048, "seed": 1}, timeout=600).json()
m = r["choices"][0]["message"]
print("SMOKE", json.dumps({"probe": "tool_call", "secs": round(time.time() - t0, 1),
      "reasoning_chars": len(m.get("reasoning_content") or m.get("reasoning") or ""),
      "content_chars": len(m.get("content") or ""), "tool_calls": m.get("tool_calls"),
      "finish": r["choices"][0]["finish_reason"], "completion_tokens": r["usage"]["completion_tokens"]}), flush=True)


def one(i):
    j = requests.post(URL, json={**base, "max_tokens": 512, "seed": 100 + i}, timeout=600).json()
    return j["usage"]["completion_tokens"]


for conc in (1, 8, 32):
    t0 = time.time()
    with ThreadPoolExecutor(conc) as ex:
        toks = sum(ex.map(one, range(conc)))
    dt = time.time() - t0
    print("SMOKE", json.dumps({"probe": "throughput", "concurrency": conc, "completion_tokens": toks,
          "secs": round(dt, 1), "tok_per_s": round(toks / dt)}), flush=True)
