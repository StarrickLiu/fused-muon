"""
Simple performance benchmark: fused NS step vs PyTorch NS step.

Marked with @pytest.mark.benchmark so it can be optionally skipped via:
    pytest -m "not benchmark"
"""

import pytest
import torch
import time

# ---------------------------------------------------------------------------
# Skip when no CUDA GPU is available
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.benchmark,
]

# ---------------------------------------------------------------------------
# Import fused extension; skip if not compiled
# ---------------------------------------------------------------------------
try:
    import muon_fused._fused_muon_C as _C
    _HAS_CUDA_EXT = True
except ImportError:
    _C = None
    _HAS_CUDA_EXT = False

requires_cuda_ext = pytest.mark.skipif(
    not _HAS_CUDA_EXT, reason="fused CUDA extension not compiled"
)


# ---------------------------------------------------------------------------
# Pure-PyTorch NS step (reference)
# ---------------------------------------------------------------------------
def pytorch_ns_step(X: torch.Tensor) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    A = X @ X.T
    B = b * A + c * A @ A
    return a * X + B @ X


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def benchmark_fn(fn, warmup: int = 10, repeats: int = 50) -> float:
    """Time *fn* with CUDA synchronisation; return median time in seconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    return times[len(times) // 2]  # median


# ---------------------------------------------------------------------------
# Benchmark: fused single step vs PyTorch single step
# ---------------------------------------------------------------------------
@requires_cuda_ext
def test_fused_faster_than_pytorch_single_step():
    """Fused NS step should be at least 1.1x faster than PyTorch for 4096x4096."""
    m, n = 4096, 4096
    torch.manual_seed(0)

    G = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    X = (G / (G.norm() + 1e-7)).contiguous()

    # Benchmark fused
    fused_time = benchmark_fn(lambda: _C.newton_schulz_step(X))

    # Benchmark PyTorch
    pytorch_time = benchmark_fn(lambda: pytorch_ns_step(X))

    speedup = pytorch_time / fused_time if fused_time > 0 else float("inf")

    # Print for visibility when run with -s
    print(
        f"\n[4096x4096 single step] "
        f"fused={fused_time*1e3:.3f}ms  pytorch={pytorch_time*1e3:.3f}ms  "
        f"speedup={speedup:.2f}x"
    )

    assert speedup >= 1.1, (
        f"Fused kernel speedup {speedup:.2f}x is below the 1.1x gate "
        f"(fused={fused_time*1e3:.3f}ms, pytorch={pytorch_time*1e3:.3f}ms)"
    )


# ---------------------------------------------------------------------------
# Benchmark: fused full iteration (5 steps) vs PyTorch
# ---------------------------------------------------------------------------
@requires_cuda_ext
def test_fused_faster_than_pytorch_multi_step():
    """Fused 5-step NS iteration should be at least 1.1x faster than PyTorch."""
    m, n = 4096, 4096
    steps = 5
    torch.manual_seed(0)

    G = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)

    def fused_fn():
        return _C.fused_newton_schulz(G, steps)

    def pytorch_fn():
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G.bfloat16()
        X = X / (X.norm() + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        return X

    fused_time = benchmark_fn(fused_fn)
    pytorch_time = benchmark_fn(pytorch_fn)

    speedup = pytorch_time / fused_time if fused_time > 0 else float("inf")

    print(
        f"\n[4096x4096 x{steps} steps] "
        f"fused={fused_time*1e3:.3f}ms  pytorch={pytorch_time*1e3:.3f}ms  "
        f"speedup={speedup:.2f}x"
    )

    assert speedup >= 1.1, (
        f"Fused kernel speedup {speedup:.2f}x is below the 1.1x gate "
        f"(fused={fused_time*1e3:.3f}ms, pytorch={pytorch_time*1e3:.3f}ms)"
    )
