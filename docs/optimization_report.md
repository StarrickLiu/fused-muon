# Newton-Schulz GEMM 优化完整报告

> Muon 优化器 NS 迭代: X_{k+1} = (aI + bA + cA²) · X_k, A = X_k·X_kᵀ
>
> 系数: (a, b, c) = (3.4445, -4.7750, 2.0315)
>
> 硬件: NVIDIA A800-SXM4-80GB (SM80), BF16 Tensor Core Peak = 312 TFLOPS

---

## 1. 背景

### 1.1 Muon 优化器

**Muon** (Momentum Orthogonalized by Newton-Schulz) 对梯度更新矩阵 G，用 NS 迭代近似极分解的正交因子 UVᵀ，使更新在所有方向均匀分配。

### 1.2 NS 迭代的 GEMM 分解

五次多项式 f(x) = ax + bx³ + cx⁵，系数 (a,b,c) = (3.4445, -4.7750, 2.0315)。

每步迭代分解为 3 个 GEMM：

```
GEMM1:  A = X @ Xᵀ                    (m×n) @ (n×m) → (m×m)  对称输出
GEMM2:  B = c·(A@A) + b·A + a·I       (m×m) @ (m×m) → (m×m)  对称输出
GEMM3:  X_new = B @ X                  (m×m) @ (m×n) → (m×n)  一般输出
```

其中 m ≤ n（取较小维度），每步 4nm² + 2m³ FLOPs，严格串行。

---

## 2. 优化方案总览

| 步骤 | torch 实现 | 我们的实现 | 核心优化 |
|---|---|---|---|
| GEMM1 | cuBLAS gemm (2m²n) | CuTe SYRK 或 cuBLAS fallback | 省 50% FLOPs (对称输出) |
| GEMM2 | cuBLAS gemm + elem + diag (3 kernel) | CuTe SYRK + fused epilogue (1 kernel) | 省 50% FLOPs + 消除 2 kernel |
| GEMM3 | cuBLAS gemm | cuBLAS gemm | 标准 GEMM，已 ~80% MFU |

### GEMM1 自适应 Dispatch

```
m ≤ 1280:            cuBLAS fallback (SYRK block 数太少)
m ≥ 2048, n/m ≤ 4:   CuTe SYRK (64×64 或 128×128 tile)
5 ≤ n/m ≤ 8:         CuTe SYRK + Split-K
n/m > 8:             cuBLAS fallback
```

---

## 3. 端到端结果

### 全 Qwen Shape 每步 NS Iteration

| Model | Layer | (m, n) | G1 策略 | G1 加速 | G2 加速 | **端到端** |
|---|---|---|---|---|---|---|
| 0.5B | QKV | (896, 1152) | cuBLAS | 1.05x | 1.24x | **1.09x** |
| | O | (896, 896) | cuBLAS | 1.03x | 1.22x | **1.09x** |
| | Gate+Up | (896, 9728) | cuBLAS | 1.01x | 1.27x | **1.03x** |
| | Down | (896, 4864) | cuBLAS | 1.00x | 1.27x | **1.04x** |
| VL-3B | QKV | (1280, 3840) | cuBLAS | 1.01x | 1.16x | **1.03x** |
| | O | (1280, 1280) | cuBLAS | 1.03x | 1.17x | **1.07x** |
| **3B** | QKV | (2048, 2560) | SYRK-64 | **1.59x** | 1.69x | **1.38x** |
| | O | (2048, 2048) | SYRK-64 | **1.56x** | 1.70x | **1.36x** |
| | Gate+Up | (2048, 22016) | cuBLAS | 1.00x | 1.69x | **1.04x** |
| | Down | (2048, 11008) | Split-K | 1.39x | 1.69x | **1.22x** |
| **Q3-4B** | QKV | (2560, 6144) | SYRK-128 | **1.72x** | 1.64x | **1.32x** |
| | O | (2560, 4096) | SYRK-128 | **1.60x** | 1.62x | **1.33x** |
| | Gate+Up | (2560, 19456) | Split-K | 1.46x | 1.65x | **1.21x** |
| | Down | (2560, 9728) | SYRK-128 | **1.80x** | 1.64x | **1.31x** |
| **7B** | QKV | (3584, 4608) | SYRK-128 | **1.67x** | 1.78x | **1.38x** |
| | O | (3584, 3584) | SYRK-128 | **1.61x** | 1.78x | **1.39x** |
| | Gate+Up | (3584, 37888) | cuBLAS | 1.00x | 1.78x | **1.03x** |
| | Down | (3584, 18944) | Split-K | 1.48x | 1.78x | **1.24x** |
| **Std** | | (4096, 4096) | SYRK-128 | **1.66x** | **1.84x** | **1.42x** |

**19/19 shape ≥ 1.03x。核心 shape (m≥2048, n/m≤4) 稳定 1.31-1.42x。**

---

## 4. GEMM1 优化: SYRK

### 4.1 原理

X@Xᵀ 结果对称，只需计算下三角 tile，省 ~50% FLOPs。SYRK kernel 通过 `linear_to_lower_tri` 映射 1D grid 到下三角 tile 坐标，跳过上三角 block。

### 4.2 最优配置

```
Tile:       128×128×32 (m ≥ 2560) / 64×64×32 (m ≤ 2048)
Pipeline:   4 stages (128-tile) / 3 stages (64-tile)
G2S:        SM80_CP_ASYNC_CACHEGLOBAL (BYPASS L1)
Swizzle:    Swizzle<2,3,3> + RowMajor<_8, _32>
MMA:        SM80_16x8x16_F32BF16BF16F32_TN, 4 warps
```

### 4.3 关键发现: BYPASS L1

工作集 (64KB) >> L1 cache (28KB)，`cp.async.ca` 白白增加 bank 压力。切换到 `cp.async.cg` (BYPASS L1) 性能提升 **56%** (491 → 314 us at m=4096)。

CUTLASS 2.x 对 K=32 自动使用 BYPASS，NCU source 级确认 `LDGSTS.E.BYPASS.LTC128B.128`。

### 4.4 双写 Epilogue

输出下三角 + 镜像写上三角，生成完整对称矩阵供 GEMM2 使用。

- 正写: CuTe vectorized `copy(out, tCgC)` — coalesced 128-bit store
- 镜像写: shared memory 转置 → coalesced 写出
- 额外开销: **+2 us** (m=4096)

### 4.5 Split-K (n/m > 5)

n >> m 时 SYRK block 数 = Mt(Mt+1)/2（仅由 m 决定），SM 利用率不足。

Split-K 沿 K 维度拆分 block：
- 两阶段: partial kernel (f32 workspace, coalesced layout) → reduce + epilogue
- (2048, 11008): GEMM1 从 1.16x → **1.39x**
- (2560, 19456): GEMM1 从 1.55x → **1.46x** (split-K 触发)

### 4.6 Tile 选择规则

| m | Tile | blocks (下三角) | 原因 |
|---|---|---|---|
| ≤ 1280 | cuBLAS fallback | — | tiles 太少 |
| 2048 | 64×64 | 528 | block 数优势 |
| ≥ 2560 | 128×128 | 210+ | 算术强度优势 (128 vs 64) |

---

## 5. GEMM2 优化: SYRK + Fused Epilogue

### 5.1 设计

A 对称 → A@A = A@Aᵀ → SYRK。epilogue 直接 fuse `D = bf16(c·acc + b·A[i][j] + a·(i==j))`，消除 torch 的 3 个 kernel (gemm + elem_scale_add + diag_add)。

### 5.2 寄存器压力

128×128 tile: 128 acc regs + 128 src load regs = 256 > 254 reg limit。
编译器自动处理 (254 regs, 0 spill)，occupancy 仍为 2 blocks/SM。
64×64 tile: 32 acc + 32 src = 64，无压力 (90 regs)。

**关键**: 两种 tile 必须分文件编译。同一文件编译会导致编译器交叉优化，64-tile 也膨胀到 254 regs。

### 5.3 NCU 验证

GEMM2 vs GEMM1 (m=4096):
- TC Util: 71.6% vs 74.3% (仅降 2.7%)
- DRAM Read: +0.2 MB (src 读取被 L2 cache 吸收)
- **Epilogue 总开销: ~9 us** (2.8%)

---

## 6. GEMM3: cuBLAS

X_new = B@X 是标准 GEMM（输出不对称），无 SYRK 优势。

cuBLAS vs CUTLASS 2.x 对比显示 cuBLAS 在大部分 shape 上更优或持平 (75-82% MFU)。直接使用 cuBLAS。

---

## 7. 性能瓶颈分析

### 7.1 中等 m (2048-2560) 加速比 < 2x

NCU 对比 m=4096 vs m=2560:

| 指标 | m=4096 | m=2560 |
|---|---|---|
| TC Util | 71.5% | **55.8%** |
| Active Warps | 7.35 | **6.86** |
| math_throttle | 28.7% | **23.8%** |

SYRK 只有 210 blocks (m=2560)，108 SM × 2 blocks/SM = 216 容量，6 个 SM 只分到 1 block → warp 并发度不足 → MMA 喂不饱。

### 7.2 极端 n/m 时 SYRK vs cuBLAS

CUTLASS 2.x RankK 不支持 split-K。对 (3584, 37888) 的对比:

| 方案 | Time | 说明 |
|---|---|---|
| cuBLAS full GEMM | 3682 us | SM 利用率高 (896 blocks) |
| Our SYRK (无 split-K) | 4093 us | 406 blocks |
| Our SYRK split-K=2 | 3798 us | 812 blocks，接近 cuBLAS |
| CUTLASS GEMM best | 4203 us | 比 cuBLAS 慢 14% |
| CUTLASS RankK | 4617 us | 不支持 split-K |

Our split-K SYRK 和 cuBLAS 差距 ~3%，主因是 mainloop 的 mio_throttle (28.6% vs CUTLASS 的 17.3%)。

### 7.3 mio_throttle 差距

| 指标 | Our SYRK | CUTLASS GEMM |
|---|---|---|
| mio_throttle | **25-29%** | **17%** |
| math_throttle | 7-8% | 13% |

我们的 smem→register 数据流效率低于 CUTLASS。CUTLASS 使用 WarpRakedThreadMap + TensorOpMultiplicandCrosswise 布局，smem 访问模式更优。这是剩余 ~6% 性能差距的根因。

---

## 8. 测量方法论

### GPU DVFS
A800 空闲后降频到 210 MHz (max 1410 MHz)。benchmark 需 **≥5 秒连续 warmup**。最佳实践: 同一程序内连续跑多个 shape，或先跑大 kernel 预热。

### NCU 偏差
NCU gpu__time_duration 有 ~60-70 us 固定开销。绝对时间以 cudaEvent 为准。

### 分文件编译
128×128 和 64×64 kernel 分 .cu 文件。同文件编译导致编译器全局寄存器分配，64-tile 从 90 膨胀到 254 regs。

---

## 9. 代码结构

```
src/ns_gemm/                           # 工业级实现
├── ns_gemm.h                          # 公共 API
├── ns_gemm.cu                         # Dispatcher (tile/split-K/cuBLAS 选择)
├── syrk_common.cuh                    # 统一 SYRK kernel 模板
├── syrk_splitk.cuh                    # Split-K partial + reduce kernel
├── syrk_128.cu                        # 128×128 tile (234/254 regs)
├── syrk_64.cu                         # 64×64 tile (90/92 regs)
├── test_ns_gemm.cu                    # 正确性 + 全量 benchmark
├── profile_splitk.cu                  # Split-K NCU profiling
└── profile_cutlass_sweep.cu           # CUTLASS config 对比

benchmarks/ns_gemm_profile/            # 研究代码
├── syrk_final_sweep.cu                # GEMM1 参数扫描
├── syrk_annotated.cu                  # CuTe SYRK 详细注释
├── gemm2_solo.cu / gemm2_64_solo.cu   # GEMM2 独立编译
├── gemm2_fullbench.cu                 # GEMM2 全尺寸 benchmark
├── gemm3_benchmark.cu                 # GEMM3 对比
├── syrk_mirror_vs_dualwrite.cu        # 双写 vs mirror 对比
├── gemm2_wave_opt.cu                  # Split-K/Persistent 实验
├── gemm2_2warp.cu                     # 2-warp 实验
└── verify_s1.cu                       # 正确性验证
```

### 构建与运行

```bash
cd src/ns_gemm && mkdir -p build
nvcc -O3 -std=c++17 --generate-code=arch=compute_80,code=sm_80 \
  -I../../third/cutlass/include -lcublas \
  -o build/test_ns_gemm ns_gemm.cu syrk_128.cu syrk_64.cu test_ns_gemm.cu

# 先热 GPU
CUDA_VISIBLE_DEVICES=0 ../benchmarks/ns_gemm_profile/build/syrk_final_sweep 4096 4096 > /dev/null
CUDA_VISIBLE_DEVICES=0 ./build/test_ns_gemm
```

---

## 10. 未来优化方向

| 方向 | 预期收益 | 难度 | 说明 |
|---|---|---|---|
| WarpRaked G2S | GEMM1/2 +5-10% | 高 | 减少 mio_throttle，缩小 vs CUTLASS 差距 |
| Split-K kernel 优化 | 大 n/m +10-20% | 中 | 减少 workspace 带宽，优化 L2 locality |
| GEMM2+GEMM3 fusion | 省 m² 访存 | 高 | B 矩阵不落地 global memory |
| 多步 NS pipeline | 减少 launch | 高 | persistent kernel 跨迭代流水线 |
| Stream-K 替代 Split-K | 更好 wave 均衡 | 中 | 无 reduction 开销 |
