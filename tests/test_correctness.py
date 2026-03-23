"""
Test fused Newton-Schulz step vs pure PyTorch reference.

Validates numerical correctness across a range of matrix shapes by comparing
the fused CUDA kernel output against a straightforward PyTorch implementation.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip the entire module when no CUDA GPU is available
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# ---------------------------------------------------------------------------
# Try importing the fused CUDA extension; skip if not compiled
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
# Pure-PyTorch reference (identical to muon_fused.ns_step fallback)
# ---------------------------------------------------------------------------
def pytorch_ns_step(X: torch.Tensor) -> torch.Tensor:
    """One Newton-Schulz iteration step in pure PyTorch."""
    a, b, c = 3.4445, -4.7750, 2.0315
    A = X @ X.T
    B = b * A + c * A @ A
    return a * X + B @ X


def pytorch_ns_multi(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Full Newton-Schulz iteration in pure PyTorch (bf16)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T.contiguous()
        transposed = True
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T.contiguous()
    return X


# ---------------------------------------------------------------------------
# Test shapes
# ---------------------------------------------------------------------------
SHAPES = [
    (896, 896),
    (2048, 2048),
    (2048, 2560),
    (2560, 4096),
    (3584, 4608),
    (4096, 4096),
]


# ---------------------------------------------------------------------------
# Single-step correctness
# ---------------------------------------------------------------------------
@requires_cuda_ext
@pytest.mark.parametrize("shape", SHAPES, ids=[f"{m}x{n}" for m, n in SHAPES])
def test_single_step(shape):
    """Compare one fused NS step against PyTorch reference."""
    m, n = shape
    torch.manual_seed(42)

    # Create a normalised random bf16 matrix on GPU
    G = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    X = G / (G.norm() + 1e-7)
    X = X.contiguous()

    # Fused step
    fused_out = _C.newton_schulz_step(X)

    # PyTorch reference step
    ref_out = pytorch_ns_step(X)

    # Compare: max_abs_error / max_value < 5% (bf16 precision)
    abs_diff = (fused_out.float() - ref_out.float()).abs()
    max_abs = abs_diff.max().item()
    max_val = ref_out.float().abs().max().item()
    rel = max_abs / max(max_val, 1e-6)

    assert rel < 0.05, (
        f"Relative error {rel:.4f} (max_abs={max_abs:.4e}, max_val={max_val:.4f}) "
        f"exceeds 5% tolerance for shape {shape}"
    )


# ---------------------------------------------------------------------------
# Multi-step convergence
# ---------------------------------------------------------------------------
@requires_cuda_ext
@pytest.mark.parametrize("shape", SHAPES, ids=[f"{m}x{n}" for m, n in SHAPES])
def test_multi_step_convergence(shape):
    """Run 5 NS steps with both fused and PyTorch, compare results."""
    m, n = shape
    steps = 5
    torch.manual_seed(123)

    G = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)

    # Fused path (via the full Python wrapper which handles normalize + transpose)
    fused_out = _C.fused_newton_schulz(G, steps)

    # PyTorch reference
    ref_out = pytorch_ns_multi(G, steps)

    # Compare: max_abs_error / max_value < 5%
    abs_diff = (fused_out.float() - ref_out.float()).abs()
    max_abs = abs_diff.max().item()
    max_val = max(ref_out.float().abs().max().item(), fused_out.float().abs().max().item(), 1e-6)
    rel = max_abs / max_val

    assert rel < 0.05, (
        f"Relative error {rel:.4f} (max_abs={max_abs:.4e}, max_val={max_val:.4f}) "
        f"exceeds 5% tolerance for shape {shape} after {steps} steps"
    )


# ---------------------------------------------------------------------------
# Sanity: output shape matches input shape
# ---------------------------------------------------------------------------
@requires_cuda_ext
@pytest.mark.parametrize("shape", SHAPES, ids=[f"{m}x{n}" for m, n in SHAPES])
def test_output_shape(shape):
    """Ensure fused NS preserves the matrix shape."""
    m, n = shape
    G = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    out = _C.fused_newton_schulz(G, 5)
    assert out.shape == G.shape, f"Expected {G.shape}, got {out.shape}"
