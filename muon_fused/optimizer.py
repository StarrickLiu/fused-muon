"""
FusedMuon -- Muon optimizer with fused Newton-Schulz orthogonalisation.

This is a drop-in replacement for the Muon optimizer that leverages a fused
CUDA kernel (when available) for the Newton-Schulz iteration, reducing kernel-
launch overhead and improving training throughput.  When the CUDA extension is
not installed the optimizer transparently falls back to a pure-PyTorch
implementation.

Reference:
    * Muon -- MomentUm Orthogonalized by Newton-schulz
      https://github.com/KellerJordan/Muon
    * Moonlight (MoonshotAI)
      https://github.com/MoonshotAI/Moonlight
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import Optimizer

from muon_fused.ns_step import fused_newton_schulz, workspace_size


class FusedMuon(Optimizer):
    """Muon optimizer with fused Newton-Schulz orthogonalisation.

    Internally runs SGD with momentum on each parameter, then replaces the
    update for every 2-D parameter with its nearest orthogonal matrix (computed
    via Newton-Schulz iteration).  Parameters that are not suitable for Muon
    (1-D biases, embeddings, etc.) are optimised with AdamW instead.

    Parameter groups must contain a boolean flag ``use_muon`` that selects the
    algorithm for each group.

    Args:
        param_groups: Iterable of parameter groups.  Each group is a dict that
            **must** include ``"params"`` and ``"use_muon"`` keys.
        lr: Base learning rate (default ``1e-3``).
        wd: Weight decay coefficient (default ``0.1``).
        momentum: SGD momentum factor for Muon groups (default ``0.95``).
        nesterov: Use Nesterov momentum (default ``True``).
        ns_steps: Number of Newton-Schulz iterations (default ``5``).
        adamw_betas: Beta coefficients for AdamW groups (default ``(0.9, 0.95)``).
        adamw_eps: Epsilon for AdamW groups (default ``1e-8``).
    """

    def __init__(
        self,
        param_groups,
        lr: float = 1e-3,
        wd: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        defaults: dict[str, Any] = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(param_groups, defaults)

        # Cache of pre-allocated workspaces keyed by (m, n) shape.
        self._workspaces: dict[tuple[int, int], torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def adjust_lr_for_muon(lr: float, param_shape: torch.Size) -> float:
        """Scale learning rate by sqrt(max(fan_in, fan_out)) as in the paper."""
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return lr * adjusted_ratio

    def _get_workspace(self, m: int, n: int, device: torch.device) -> torch.Tensor:
        """Return a cached workspace buffer, allocating one if necessary."""
        key = (m, n)
        if key not in self._workspaces:
            ws_numel = workspace_size(m, n)
            self._workspaces[key] = torch.empty(
                ws_numel, dtype=torch.bfloat16, device=device,
            )
        return self._workspaces[key]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimisation step.

        Args:
            closure: An optional closure that re-evaluates the model and
                returns the loss.

        Returns:
            The loss value if *closure* was provided, otherwise ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon: bool = group.get("use_muon", False)
            lr: float = group["lr"]
            wd: float = group["wd"]

            if use_muon:
                # ----------------------------------------------------------
                # Muon path: SGD-momentum + Newton-Schulz orthogonalisation
                # ----------------------------------------------------------
                momentum: float = group["momentum"]
                nesterov: bool = group["nesterov"]
                ns_steps: int = group["ns_steps"]

                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue

                    # Flatten >2-D parameters to 2-D
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    state = self.state[p]

                    # Lazy-init momentum buffer
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)

                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf

                    # Fused Newton-Schulz orthogonalisation
                    workspace = self._get_workspace(
                        g.size(0), g.size(1), g.device,
                    )
                    u = fused_newton_schulz(g, steps=ns_steps, workspace=workspace)

                    # Reshape back to original parameter shape if it was >2D
                    if u.shape != p.shape:
                        u = u.view_as(p)

                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                    p.data.mul_(1 - lr * wd)
                    p.data.add_(u, alpha=-adjusted_lr)

            else:
                # ----------------------------------------------------------
                # AdamW fallback for non-Muon parameters
                # ----------------------------------------------------------
                beta1, beta2 = group["adamw_betas"]
                eps: float = group["adamw_eps"]

                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue

                    state = self.state[p]

                    # Lazy-init AdamW state
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)

                    state["step"] += 1
                    step: int = state["step"]
                    buf1 = state["moment1"]
                    buf2 = state["moment2"]

                    buf1.lerp_(g, 1 - beta1)
                    buf2.lerp_(g.square(), 1 - beta2)

                    g = buf1 / (eps + buf2.sqrt())

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    scale = bias_correction1 / bias_correction2 ** 0.5

                    p.data.mul_(1 - lr * wd)
                    p.data.add_(g, alpha=-lr / scale)

        return loss
