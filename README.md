<p align="center">
  <h1 align="center">⚡ fused-muon</h1>
  <p align="center">
    <b>Fused CUDA kernels for the Muon optimizer's Newton-Schulz iteration</b>
  </p>
  <p align="center">
    <a href="https://github.com/StarrickLiu/fused-muon/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"/></a>
    <a href="https://github.com/StarrickLiu/fused-muon"><img alt="CUDA" src="https://img.shields.io/badge/CUDA-SM80%2B-green.svg"/></a>
    <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E%3D2.0-red.svg"/></a>
    <a href="https://github.com/StarrickLiu/fused-muon"><img alt="BF16" src="https://img.shields.io/badge/precision-BF16-orange.svg"/></a>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-benchmarks">Benchmarks</a> •
    <a href="#-how-it-works">How It Works</a> •
    <a href="README_CN.md">中文文档</a> •
    <a href="docs/optimization_report.md">Optimization Report</a>
  </p>
</p>

---

Drop-in replacement for the [Muon optimizer](https://github.com/KellerJordan/Muon) that accelerates the Newton-Schulz orthogonalization by exploiting **matrix symmetry** with custom CuTe SYRK kernels. Achieves **up to 1.4x faster** per-step NS iteration with **identical training dynamics**.

<p align="center">
  <img src="benchmarks/figures/optimizer_step_time.png" width="600"/>
</p>

## ✨ Highlights

- 🔺 **SYRK symmetry exploitation** — `X @ Xᵀ` is symmetric, compute only the lower triangle → **50% FLOP savings**
- 🔗 **Fused GEMM epilogue** — `c·A² + b·A + a·I` in a single kernel, eliminates 2 extra kernel launches
- 🎯 **Adaptive dispatch** — 64/128 tile, Split-K, cuBLAS fallback, auto-selected per shape
- 🔌 **Drop-in API** — `from muon_fused import FusedMuon`, identical interface to standard Muon
- 🛡️ **Graceful fallback** — automatically falls back to pure PyTorch when CUDA extension is unavailable

---

## 🚀 Quick Start

### Installation

```bash
git clone --recursive https://github.com/StarrickLiu/fused-muon.git
cd fused-muon
pip install -e .
```

### Usage

```python
from muon_fused import FusedMuon

# Drop-in replacement — same API as standard Muon
optimizer = FusedMuon(model.parameters(), lr=0.02, momentum=0.95, ns_steps=5)
```

Or use just the optimized Newton-Schulz function in your own optimizer:

```python
from muon_fused import fused_newton_schulz

# Replace zeropower_via_newtonschulz5() with this
X_ortho = fused_newton_schulz(G, steps=5)
```

---

## 📊 Benchmarks

### CIFAR-10 Training: FusedMuon vs VanillaMuon vs AdamW

<table>
<tr>
<td><img src="benchmarks/figures/train_loss_vs_epoch.png" width="400"/></td>
<td><img src="benchmarks/figures/test_accuracy_vs_epoch.png" width="400"/></td>
</tr>
<tr>
<td><img src="benchmarks/figures/train_loss_vs_time.png" width="400"/></td>
<td><img src="benchmarks/figures/optimizer_step_time.png" width="400"/></td>
</tr>
</table>

> FusedMuon and VanillaMuon produce **identical training curves** (loss & accuracy overlap perfectly), confirming numerical equivalence. FusedMuon achieves measurably faster optimizer steps on this small-scale benchmark. The per-step NS iteration speedup (1.3–1.4x on large shapes) is the primary optimization target.

### Newton-Schulz Step Speedup (5 iterations)

Tested on NVIDIA A800-SXM4-80GB, BF16:

| Shape (m, n) | Vanilla (ms) | Fused (ms) | Speedup |
|:---:|:---:|:---:|:---:|
| (896, 1152) | 0.58 | 0.31 | **1.87x** |
| (896, 896) | 0.59 | 0.27 | **2.17x** |
| (2048, 2560) | 2.26 | 1.44 | **1.57x** |
| (2048, 2048) | 2.12 | 1.34 | **1.58x** |
| (2560, 4096) | 3.57 | 2.36 | **1.51x** |
| (3584, 4608) | 7.93 | 5.26 | **1.51x** |
| (4096, 4096) | 9.58 | 6.30 | **1.52x** |

### End-to-End NS Iteration Speedup (Qwen Model Shapes)

| Model | Layer | Shape (m, n) | GEMM1 | GEMM2 | E2E Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen 3B | QKV | (2048, 2560) | 1.59x | 1.69x | **1.38x** |
| Qwen3-4B | QKV | (2560, 6144) | 1.72x | 1.64x | **1.32x** |
| Qwen3-4B | Down | (2560, 9728) | 1.80x | 1.64x | **1.31x** |
| Qwen 7B | QKV | (3584, 4608) | 1.67x | 1.78x | **1.38x** |
| Qwen 7B | O | (3584, 3584) | 1.61x | 1.78x | **1.39x** |
| Standard | — | (4096, 4096) | 1.66x | 1.84x | **1.42x** |

> All 19 tested shapes achieve **≥ 1.03x** end-to-end speedup. Core shapes (m ≥ 2048, n/m ≤ 4) achieve stable **1.31–1.42x**.

---

## 🔬 How It Works

Each Newton-Schulz iteration step computes:

$$X_{k+1} = \left(aI + bA + cA^2\right) X_k, \quad A = X_k X_k^\top$$

with coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$. This decomposes into 3 GEMMs:

```
GEMM1: A = X @ Xᵀ           → CuTe SYRK (50% FLOPs saved via symmetry)
GEMM2: B = c·A² + b·A + a·I → CuTe SYRK + fused polynomial epilogue (1 kernel)
GEMM3: X_new = B @ X         → cuBLAS (standard GEMM, ~80% MFU)
```

### Key Kernel Optimizations

| Technique | Applied To | Benefit |
|:---|:---|:---|
| **SYRK lower-triangle** | GEMM1, GEMM2 | 50% compute reduction (only lower triangle tiles) |
| **Fused polynomial epilogue** | GEMM2 | Merges `c·acc + b·A + a·I` into SYRK kernel, eliminates 2 extra kernels |
| **Dual-write epilogue** | GEMM1, GEMM2 | Outputs full symmetric matrix via smem transpose, +2μs overhead |
| **BYPASS L1** (`cp.async.cg`) | All SYRK | Avoids L1 cache pollution for large working sets, **+56% speedup** |
| **Split-K** | GEMM1 (n ≫ m) | Splits K-dimension across blocks for better SM utilization |
| **Adaptive tile dispatch** | All | 64×64 tile for small m (more blocks), 128×128 for large m (higher arithmetic intensity) |

### Adaptive GEMM1 Dispatch

```
m ≤ 1280:            cuBLAS fallback (too few SYRK blocks)
m ≥ 2048, n/m ≤ 4:   CuTe SYRK 64×64 or 128×128
5 ≤ n/m ≤ 8:         CuTe SYRK 128×128 + Split-K
n/m > 8:             cuBLAS fallback (extreme aspect ratio)
```

> See [docs/optimization_report.md](docs/optimization_report.md) for full NCU profiling data, register pressure analysis, and detailed optimization history.

---

## 📦 Advanced Usage

### Parameter Groups (Recommended for LLM Training)

```python
from muon_fused import FusedMuon

# 2D+ params → Muon (SGD momentum + NS orthogonalization)
# 1D params (bias, norm) → AdamW fallback
param_groups = [
    {"params": [p for p in model.parameters() if p.ndim >= 2], "use_muon": True},
    {"params": [p for p in model.parameters() if p.ndim < 2], "use_muon": False},
]
optimizer = FusedMuon(param_groups, lr=0.02, momentum=0.95, ns_steps=5)
```

### Standalone Newton-Schulz Function

```python
from muon_fused import fused_newton_schulz

# Use in any custom optimizer
# Handles: bf16 cast, normalization, transpose (if m > n), multi-step iteration
X_ortho = fused_newton_schulz(gradient_matrix, steps=5)
```

### Pre-allocated Workspace

```python
from muon_fused.ns_step import workspace_size
import torch

# For repeated calls with the same shape, pre-allocate workspace to avoid cudaMalloc
m, n = 4096, 4096
ws = torch.empty(workspace_size(m, n), dtype=torch.uint8, device="cuda")
X_ortho = fused_newton_schulz(G, steps=5, workspace=ws)
```

---

## ⚙️ Requirements

| Requirement | Version |
|:---|:---|
| NVIDIA GPU | **SM80+** (A100, A800, H100, H200) |
| PyTorch | ≥ 2.0 |
| CUDA Toolkit | ≥ 11.8 |
| CUTLASS | Included as git submodule (header-only) |

### Installation

```bash
git clone --recursive https://github.com/StarrickLiu/fused-muon.git
cd fused-muon
pip install -e .
```

If CUTLASS submodule is missing:
```bash
git submodule update --init --recursive
```

---

## 🧪 Testing

```bash
# Run all tests (correctness + optimizer equivalence + performance)
pytest tests/ -v

# Skip performance benchmarks in CI
pytest tests/ -v -m "not benchmark"
```

## 📈 Reproduce Benchmarks

```bash
# Newton-Schulz step breakdown (per-GEMM timing)
python benchmarks/bench_ns_step.py

# CIFAR-10 training comparison (FusedMuon vs Muon vs AdamW)
python benchmarks/train_cifar10.py

# Generate figures
python benchmarks/plot_results.py
```

---

## 📚 Citation

```bibtex
@software{fused_muon,
  title  = {Fused Muon: CUDA-Optimized Newton-Schulz Iteration for the Muon Optimizer},
  author = {Xingchen Liu},
  year   = {2025},
  url    = {https://github.com/StarrickLiu/fused-muon}
}
```

## 🙏 Acknowledgments

- [Muon optimizer](https://github.com/KellerJordan/Muon) by Keller Jordan — the original Muon algorithm
- [CUTLASS](https://github.com/NVIDIA/cutlass) & [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute) by NVIDIA — tensor core abstraction
- [Moonlight](https://github.com/MoonshotAI/Moonlight) by Moonshot AI — distributed Muon implementation reference

## License

[Apache 2.0](LICENSE)
