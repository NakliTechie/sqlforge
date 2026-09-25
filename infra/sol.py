"""Speed-of-light (SoL) estimates for rollouts and GRPO training — a floor, never a forecast.

Roofline per phase, no fitted parameters (the Quail approach, github.com/fsdatalab/quail docs/architecture/planning):
- decode step (B sequences) = max(weight bytes / bandwidth, 2·P·B / FLOPs)   → tokens/s = B / step
- prefill                   = 2·P per token / FLOPs                           (compute-bound)
- LoRA train                = 8·P per trajectory token / FLOPs  (fwd 2 + bwd 4 + checkpoint recompute 2)
Ignored, so every number is optimistic: KV-cache reads (small for Qwen3.5's hybrid attention at ≤32K context and
small batch, large at big batch), attention FLOPs, host overhead, tool-call latency, spot preemption.

Real systems land well below SoL; `--eff` scales the SoL to a planning number (vLLM decode ≈ 0.4–0.6; HF generate
on MPS measured ≈ 0.2 — see `--measured`).

Usage:
  uv run python infra/sol.py                                  # climb 1 (4B bf16, batch 64) on every device
  uv run python infra/sol.py --model 2b --precision q4 --batch 1
  uv run python infra/sol.py --measured 34 --device m4pro --model 0.8b --precision bf16 --batch 1
"""
from __future__ import annotations

import argparse

# name: (label, bandwidth B/s, dense bf16 FLOP/s or None, $/h or None, source)
DEVICES = {
    "m4pro":   ("MacBook M4 Pro 24 GB (this laptop)", 273e9, 9.2e12, 0.0,
                "Apple spec (bandwidth); FLOPs derived 20 cores×128 ALU×2×1.8 GHz shader fp32"),
    "m4max":   ("Mac Studio M4 Max 64 GB (Om)", 546e9, 18.4e12, 0.0,
                "Apple spec (bandwidth); FLOPs derived 40 cores×128×2×1.8 GHz"),
    "m5ultra": ("Mac Studio M5 Ultra 96 GB (planned)", 1.2e12, 36.9e12, 0.0,
                "Apple spec 2026-08 (bandwidth); FLOPs derived 80 cores×128×2×1.8 GHz, GPU Neural Accelerators excluded"),
    "l4":      ("GCP L4 24 GB, g2-standard-4 spot", 300e9, 121e12, 0.376,
                "NVIDIA L4 datasheet (242 TF sparse, halved); price SkyPilot catalog"),
    "rtxpro":  ("GCP RTX PRO 6000 96 GB, g4-standard-48 spot", 1.597e12, 0.5e15, 1.72,
                "quail/specs rtx_pro_6000_blackwell_server.py; price GCP billing catalog 2026-09-25 (GPU+host)"),
    "a100":    ("A100 80 GB, PrimeIntellect", 2.039e12, 312e12, 1.12,
                "NVIDIA A100 datasheet; price ~/Code/infra/gpu-providers.md"),
    "h100":    ("H100 SXM 80 GB, Modal", 3.35e12, 0.9895e15, 3.95,
                "quail/specs h100_sxm.py"),
}
PARAMS = {"0.8b": 0.8e9, "2b": 2.0e9, "4b": 4.0e9}      # nominal Qwen3.5 sizes
BYTES = {"bf16": 2.0, "fp8": 1.0, "q4": 0.5625}         # q4 ≈ Ollama Q4_K_M, 4.5 bits/weight


def decode_tps(bw, flops, p, bytes_w, batch):
    step = max(p * bytes_w / bw, 2 * p * batch / flops)
    return batch / step


def climb_hours(bw, flops, p, bytes_w, a):
    """Rollout + training SoL seconds for the whole climb, in hours."""
    eps = a.episodes
    gen = eps * a.gen_tokens / decode_tps(bw, flops, p, bytes_w, a.batch)
    prefill_tok = a.traj_tokens if a.prefix_cache else a.traj_tokens * a.turns / 2  # no cache: re-read every turn
    pre = eps * prefill_tok * 2 * p / flops
    train = eps * a.traj_tokens * 8 * p / flops
    return gen / 3600, pre / 3600, train / 3600


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="4b", choices=PARAMS)
    ap.add_argument("--precision", default="bf16", choices=BYTES)
    ap.add_argument("--batch", type=int, default=64, help="concurrent sequences in decode")
    ap.add_argument("--episodes", type=int, default=13_070, help="climb 1: 200 steps × 8 tasks × 8 + 9 evals × 30")
    ap.add_argument("--gen-tokens", type=int, default=3_000, help="generated tokens per episode (estimate; calibrate from runs/<tag>/ traces)")
    ap.add_argument("--traj-tokens", type=int, default=8_000, help="final trajectory length per episode (estimate)")
    ap.add_argument("--turns", type=int, default=16, help="turns per episode, for the no-prefix-cache prefill cost")
    ap.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false",
                    help="each turn re-prefills the whole context (today's train/rollout.py HF path)")
    ap.add_argument("--eff", type=float, default=0.4, help="fraction of SoL assumed for the planning column")
    ap.add_argument("--measured", type=float, help="measured decode tokens/s; prints efficiency vs SoL for --device")
    ap.add_argument("--device", default="m4pro", choices=DEVICES)
    a = ap.parse_args()
    p, bw_bytes = PARAMS[a.model], BYTES[a.precision]

    if a.measured is not None:
        label, bw, flops, _, src = DEVICES[a.device]
        sol = decode_tps(bw, flops, p, bw_bytes, a.batch)
        print(f"{label} · {a.model} {a.precision} · batch {a.batch}")
        print(f"SoL decode {sol:,.0f} tok/s · measured {a.measured:,.0f} tok/s · efficiency {a.measured / sol:.0%}")
        print(f"source: {src}")
        return

    print(f"Qwen3.5-{a.model} {a.precision} · decode batch {a.batch} · {a.episodes:,} episodes × {a.gen_tokens:,} gen tokens"
          f" · prefix cache {'on' if a.prefix_cache else 'OFF'} · planning column at {a.eff:.0%} of SoL\n")
    hdr = f"{'device':<44}{'decode tok/s':>13}{'batch tok/s':>12}{'rollout h':>10}{'train h':>9}{'SoL h':>8}{'plan h':>8}{'plan $':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label, bw, flops, usd, _ in DEVICES.values():
        one = decode_tps(bw, flops, p, bw_bytes, 1)
        many = decode_tps(bw, flops, p, bw_bytes, a.batch)
        g, pre, tr = climb_hours(bw, flops, p, bw_bytes, a)
        sol = g + pre + tr
        plan = sol / a.eff
        cost = "free" if not usd else f"{plan * usd:,.0f}"
        print(f"{label:<44}{one:>13,.0f}{many:>12,.0f}{g + pre:>10,.1f}{tr:>9,.1f}{sol:>8,.1f}{plan:>8,.1f}{cost:>8}")
    print("\nSoL h = floor. plan h = SoL / eff. Mac FLOPs are derived shader rates (see DEVICES sources); "
          "training on Apple silicon via PyTorch MPS reaches a small share of them.")


if __name__ == "__main__":
    main()
