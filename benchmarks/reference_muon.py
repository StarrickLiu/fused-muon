"""
Vanilla Muon optimizer -- standalone reference implementation.

Extracted from wall-x for benchmarking purposes.
Source: wall_x/trainer/optimizer/muon.py
"""

import math
import torch
from torch.optim import Optimizer


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients are selected to maximize the
    slope at zero.  The result is not exactly UV^T but rather something like
    US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns
    out not to hurt model performance relative to UV^T.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an
    orthogonalization post-processing step, in which each 2D parameter's
    update is replaced with the nearest orthogonal matrix.  To efficiently
    orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Arguments:
        param_groups: Parameter groups (must contain ``use_muon`` flag).
        lr: Learning rate (0.02 is a good default for Muon params).
        wd: Weight decay.
        momentum: Momentum for the internal SGD (0.95 is a good default).
        nesterov: Whether to use Nesterov-style momentum.
        ns_steps: Number of Newton-Schulz iterations (5-6 is enough).
        adamw_betas: Betas for the internal AdamW fallback.
        adamw_eps: Epsilon for the internal AdamW fallback.
    """

    def __init__(
        self,
        param_groups,
        lr=1e-3,
        wd=0.1,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(param_groups, defaults)

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = group.get("use_muon", False)
            lr = group["lr"]
            wd = group["wd"]

            if use_muon:
                momentum = group["momentum"]
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)

                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf

                    u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    if u.shape != p.shape:
                        u = u.view_as(p)
                    adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                    p.data.mul_(1 - lr * wd)
                    p.data.add_(u, alpha=-adjusted_lr)
            else:
                beta1, beta2 = group["adamw_betas"]
                eps = group["adamw_eps"]
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue

                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)

                    state["step"] += 1
                    step = state["step"]
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
