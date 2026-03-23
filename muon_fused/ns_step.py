"""
Newton-Schulz iteration for computing the orthogonal projection of a matrix.

This module provides a fused Newton-Schulz implementation that attempts to use
a CUDA extension for maximum performance, falling back to a pure PyTorch
implementation when the extension is unavailable.
"""

import torch

# ---------------------------------------------------------------------------
# Try to load the compiled CUDA extension
# ---------------------------------------------------------------------------
try:
    import muon_fused._fused_muon_C as _C
    _HAS_CUDA_EXT = True
except ImportError:
    _C = None
    _HAS_CUDA_EXT = False


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback
# ---------------------------------------------------------------------------
def _pytorch_newton_schulz(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Pure-PyTorch Newton-Schulz iteration (bfloat16, runs on any device).

    Computes an approximate orthogonal factor of *G* using a quintic iteration
    whose coefficients are chosen to maximise the slope at zero.  The result is
    not exactly UV^T but something close enough for Muon-style optimisation.

    Args:
        G: 2-D gradient tensor.
        steps: Number of Newton-Schulz iterations (5-6 is typically enough).

    Returns:
        Approximate orthogonal matrix with the same shape as *G*.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


# ---------------------------------------------------------------------------
# Workspace helper
# ---------------------------------------------------------------------------
def workspace_size(m: int, n: int) -> int:
    """Return the number of elements needed for the CUDA workspace buffer.

    The workspace stores intermediate matrices used by the fused kernel.  When
    no CUDA extension is loaded the function still returns a reasonable size so
    that callers can pre-allocate without branching.

    Args:
        m: Number of rows of the gradient matrix.
        n: Number of columns of the gradient matrix.

    Returns:
        Number of ``bfloat16`` elements required.
    """
    if _HAS_CUDA_EXT and hasattr(_C, "workspace_size"):
        return _C.workspace_size(m, n)
    # Fallback estimate: space for two (k x k) temporaries where k = min(m, n)
    k = min(m, n)
    return 2 * k * k


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fused_newton_schulz(
    G: torch.Tensor,
    steps: int = 5,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the Newton-Schulz orthogonal projection of *G*.

    When the compiled CUDA extension (``muon_fused._fused_muon_C``) is
    available the call is dispatched to a fused kernel that performs the entire
    iteration in a single launch.  Otherwise, an equivalent pure-PyTorch
    implementation is used as a transparent fallback.

    Args:
        G: 2-D gradient tensor (any floating dtype; internally cast to bf16).
        steps: Number of Newton-Schulz iterations (default ``5``).
        workspace: Optional pre-allocated ``bfloat16`` buffer on the same
            device as *G*.  Only used by the CUDA path; ignored by the
            fallback.  Use :func:`workspace_size` to determine the required
            number of elements.

    Returns:
        Approximate orthogonal matrix with the same shape as *G*.
    """
    assert G.ndim == 2, f"Expected 2-D tensor, got {G.ndim}-D"

    # ---- CUDA fast path ----
    if _HAS_CUDA_EXT and G.is_cuda:
        try:
            if workspace is None:
                ws_numel = workspace_size(G.size(0), G.size(1))
                workspace = torch.empty(ws_numel, dtype=torch.bfloat16, device=G.device)
            return _C.fused_newton_schulz(G, steps, workspace)
        except Exception:
            # If the CUDA extension fails at runtime (e.g. unsupported shape),
            # fall through to the PyTorch path rather than crashing.
            pass

    # ---- PyTorch fallback ----
    return _pytorch_newton_schulz(G, steps)
