"""
Test FusedMuon optimizer vs vanilla Muon reference.

Creates a small MLP, clones weights to two copies, and trains each with a
different optimizer for a few steps.  The resulting parameters should stay
close (within bf16 tolerance).
"""

import pytest
import sys
import os
import copy
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Skip when no CUDA GPU is available
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# ---------------------------------------------------------------------------
# Make sure the benchmarks directory is importable
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARKS_DIR = os.path.join(REPO_ROOT, "benchmarks")
if BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, BENCHMARKS_DIR)

# ---------------------------------------------------------------------------
# Import FusedMuon; skip if the CUDA extension is not compiled
# ---------------------------------------------------------------------------
try:
    from muon_fused.optimizer import FusedMuon
    _HAS_FUSED = True
except (ImportError, ModuleNotFoundError):
    FusedMuon = None
    _HAS_FUSED = False

requires_fused = pytest.mark.skipif(
    not _HAS_FUSED, reason="FusedMuon not available (CUDA extension not compiled)"
)

# ---------------------------------------------------------------------------
# Import reference Muon from benchmarks/
# ---------------------------------------------------------------------------
try:
    from reference_muon import Muon as ReferenceMuon
    _HAS_REF = True
except (ImportError, ModuleNotFoundError):
    ReferenceMuon = None
    _HAS_REF = False

requires_ref = pytest.mark.skipif(
    not _HAS_REF, reason="reference_muon.py not found in benchmarks/"
)


# ---------------------------------------------------------------------------
# Helper: build a small 3-layer MLP
# ---------------------------------------------------------------------------
def make_mlp(device="cuda", dtype=torch.float32):
    """Return a 3-layer MLP: Linear(128,256) -> ReLU -> Linear(256,128)."""
    model = nn.Sequential(
        nn.Linear(128, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, 128, bias=False),
    ).to(device=device, dtype=dtype)
    return model


def clone_model(model):
    """Deep-copy model weights so two models start identically."""
    return copy.deepcopy(model)


# ---------------------------------------------------------------------------
# Test: FusedMuon vs reference Muon produce similar parameters
# ---------------------------------------------------------------------------
@requires_fused
@requires_ref
def test_fused_vs_reference_muon():
    """Train two identical MLPs for 5 steps: one with FusedMuon, one with
    reference Muon.  Parameters should remain close."""
    torch.manual_seed(0)
    device = "cuda"

    # Build two identical models
    model_fused = make_mlp(device=device)
    model_ref = clone_model(model_fused)

    # Gather parameters (only 2D weight matrices, matching Muon convention)
    fused_params = [p for p in model_fused.parameters() if p.ndim == 2]
    ref_params = [p for p in model_ref.parameters() if p.ndim == 2]

    lr = 0.02
    momentum = 0.95

    opt_fused = FusedMuon(fused_params, lr=lr, momentum=momentum)
    opt_ref = ReferenceMuon(ref_params, lr=lr, momentum=momentum)

    # Synthetic training loop
    steps = 5
    for step in range(steps):
        torch.manual_seed(step)  # same input each step
        x = torch.randn(32, 128, device=device)

        # Forward + backward on fused model
        loss_f = model_fused(x).pow(2).mean()
        opt_fused.zero_grad()
        loss_f.backward()
        opt_fused.step()

        # Forward + backward on reference model
        loss_r = model_ref(x).pow(2).mean()
        opt_ref.zero_grad()
        loss_r.backward()
        opt_ref.step()

    # Compare parameters
    for i, (pf, pr) in enumerate(zip(fused_params, ref_params)):
        abs_diff = (pf.float() - pr.float()).abs()
        ref_abs = pr.float().abs().clamp(min=1e-6)
        rel_diff = abs_diff / ref_abs
        max_rel = rel_diff.max().item()

        assert max_rel < 0.05, (
            f"Parameter {i}: max relative error {max_rel:.4f} exceeds 5% "
            f"tolerance after {steps} training steps"
        )


# ---------------------------------------------------------------------------
# Test: FusedMuon can run without errors (smoke test)
# ---------------------------------------------------------------------------
@requires_fused
def test_fused_muon_smoke():
    """Ensure FusedMuon runs a training step without crashing."""
    torch.manual_seed(0)
    model = make_mlp(device="cuda")
    params = [p for p in model.parameters() if p.ndim == 2]
    opt = FusedMuon(params, lr=0.02, momentum=0.95)

    x = torch.randn(16, 128, device="cuda")
    loss = model(x).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Just check we didn't crash and params are finite
    for p in params:
        assert torch.isfinite(p).all(), "Non-finite parameter after FusedMuon step"
