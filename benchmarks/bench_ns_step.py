#!/usr/bin/env python3
"""
Benchmark the Newton-Schulz orthogonalization step across all Qwen model shapes.

Compares:
  - Vanilla PyTorch NS step (torch.compile'd, from reference_muon.py)
  - Fused NS step (from muon_fused.ns_step)

Outputs a JSON log to benchmarks/logs/ns_step_benchmark.json and prints a
summary table to stdout.
"""

import json
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Resolve paths -- allow running from repo root or from benchmarks/
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Import the two implementations
# ---------------------------------------------------------------------------
from benchmarks.reference_muon import zeropower_via_newtonschulz5 as vanilla_ns

try:
    from muon_fused.ns_step import fused_newton_schulz as fused_ns
except ImportError:
    fused_ns = None
    print("[WARN] muon_fused.ns_step could not be imported; fused benchmarks will be skipped.")

# ---------------------------------------------------------------------------
# Shapes to benchmark (Qwen model dimensions)
# ---------------------------------------------------------------------------
SHAPES = [
    (896, 1152),
    (896, 896),
    (2048, 2560),
    (2048, 2048),
    (2560, 6144),
    (2560, 4096),
    (3584, 4608),
    (3584, 3584),
    (4096, 4096),
]

NS_STEPS = 5
WARMUP_ITERS = 200
MEASURE_ITERS = 50
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def bench_fn(fn, G, ns_steps, warmup, measure):
    """Time *fn(G, ns_steps)* using CUDA events. Returns median time in ms."""
    # Warmup
    for _ in range(warmup):
        _ = fn(G, ns_steps)
    torch.cuda.synchronize()

    times = []
    for _ in range(measure):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = fn(G, ns_steps)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median = times[len(times) // 2]
    return median, times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    results = []

    header = f"{'Shape':>16s}  {'Vanilla (ms)':>14s}  {'Fused (ms)':>14s}  {'Speedup':>8s}"
    print(header)
    print("-" * len(header))

    for shape in SHAPES:
        G = torch.randn(shape, dtype=DTYPE, device=device)

        # --- Vanilla ---
        vanilla_med, _ = bench_fn(vanilla_ns, G, NS_STEPS, WARMUP_ITERS, MEASURE_ITERS)

        # --- Fused ---
        if fused_ns is not None:
            fused_med, _ = bench_fn(fused_ns, G, NS_STEPS, WARMUP_ITERS, MEASURE_ITERS)
            speedup = vanilla_med / fused_med if fused_med > 0 else float("inf")
        else:
            fused_med = float("nan")
            speedup = float("nan")

        row = {
            "shape": list(shape),
            "vanilla_ms": round(vanilla_med, 4),
            "fused_ms": round(fused_med, 4) if fused_ns is not None else None,
            "speedup": round(speedup, 3) if fused_ns is not None else None,
        }
        results.append(row)

        fused_str = f"{fused_med:14.4f}" if fused_ns is not None else f"{'N/A':>14s}"
        speedup_str = f"{speedup:7.2f}x" if fused_ns is not None else f"{'N/A':>8s}"
        print(f"{str(shape):>16s}  {vanilla_med:14.4f}  {fused_str}  {speedup_str}")

    # --- Save JSON log ---
    log_dir = os.path.join(SCRIPT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "ns_step_benchmark.json")

    meta = {
        "ns_steps": NS_STEPS,
        "warmup_iters": WARMUP_ITERS,
        "measure_iters": MEASURE_ITERS,
        "dtype": str(DTYPE),
        "device": torch.cuda.get_device_name(device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    payload = {"meta": meta, "results": results}

    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {log_path}")


if __name__ == "__main__":
    main()
