#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/copy_traits_sm80.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>
#include "cutlass/bfloat16.h"

namespace xcompute {
namespace detail {

using namespace cute;
using Element = cutlass::bfloat16_t;

// NS coefficients
constexpr float NS_A = 3.4445f;   // diagonal
constexpr float NS_B = -4.7750f;  // linear
constexpr float NS_C = 2.0315f;   // quadratic

// ============================================================
// Block index mapping for SYRK lower triangle
// ============================================================
__device__ __forceinline__
void linear_to_lower_tri(int idx, int& row, int& col) {
  row = int((-1.0 + sqrt(1.0 + 8.0 * double(idx))) / 2.0);
  while (row * (row + 1) / 2 > idx) --row;
  while ((row + 1) * (row + 2) / 2 <= idx) ++row;
  col = idx - row * (row + 1) / 2;
}

// ============================================================
// Shared memory storage for A and B panels
// ============================================================
template <class TA, class TB, class SA, class SB>
struct SmemStorage {
  cute::ArrayEngine<TA, cute::cosize_v<SA>> A;
  cute::ArrayEngine<TB, cute::cosize_v<SB>> B;
};

// ============================================================
// Unified SYRK kernel template
// EpilogueMode: 0 = GEMM1 (D=bf16(acc)), 1 = GEMM2 (D=bf16(c*acc+b*src+a*diag))
// TILE_M: 64 or 128
// ============================================================
template <int EpilogueMode, int TILE_M,
          class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class CopyA, class S2RA,
          class TB, class BStride, class BSmemLayout, class CopyB, class S2RB,
          class TC, class CStride, class CSmemLayout, class MMA>
__global__ static __launch_bounds__(decltype(size(MMA{}))::value)
void syrk_kernel(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A_ptr, AStride dA, ASmemLayout, CopyA copy_a, S2RA,
                 TB const* B_ptr, BStride dB, BSmemLayout, CopyB copy_b, S2RB,
                 TC* C_ptr, CStride dC, CSmemLayout, MMA mma, int M_tiles,
                 TC const* src_ptr /* only used when EpilogueMode==1 */) {
  int M = get<0>(shape_MNK);
  int i_tile, j_tile;
  linear_to_lower_tri(blockIdx.x, i_tile, j_tile);
  if (i_tile >= M_tiles) return;

  Tensor mA = make_tensor(make_gmem_ptr(A_ptr), select<0,2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B_ptr), select<1,2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), select<0,1>(shape_MNK), dC);
  Tensor gA = local_tile(mA, cta_tiler, make_coord(i_tile, _, _), Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, make_coord(_, j_tile, _), Step< X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, make_coord(i_tile, j_tile, _), Step<_1, _1, X>{});

  extern __shared__ char smem[];
  using SS = SmemStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SS& storage = *reinterpret_cast<SS*>(smem);
  Tensor sA = make_tensor(make_smem_ptr(storage.A.begin()), ASmemLayout{});
  Tensor sB = make_tensor(make_smem_ptr(storage.B.begin()), BSmemLayout{});

  ThrCopy thr_a = copy_a.get_slice(threadIdx.x);
  auto tAgA = thr_a.partition_S(gA); auto tAsA = thr_a.partition_D(sA);
  ThrCopy thr_b = copy_b.get_slice(threadIdx.x);
  auto tBgB = thr_b.partition_S(gB); auto tBsB = thr_b.partition_D(sB);
  auto K_PIPE = size<3>(tAsA);
  int k_count = size<3>(tAgA); int k_next = 0;

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  auto tCgC = thr_mma.partition_C(gC);
  auto tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
  auto tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
  auto tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);
  auto s2r_a = make_tiled_copy_A(S2RA{}, mma);
  auto s2r_b = make_tiled_copy_B(S2RB{}, mma);
  auto tXsA = s2r_a.get_slice(threadIdx.x).partition_S(sA);
  auto tXrA = s2r_a.get_slice(threadIdx.x).retile_D(tCrA);
  auto tXsB = s2r_b.get_slice(threadIdx.x).partition_S(sB);
  auto tXrB = s2r_b.get_slice(threadIdx.x).retile_D(tCrB);
  auto K_BLK = size<2>(tCrA);

  // ── Mainloop: cp.async pipeline ──
  CUTE_UNROLL
  for (int p = 0; p < K_PIPE - 1; ++p) {
    copy(copy_a, tAgA(_,_,_,k_next), tAsA(_,_,_,p));
    copy(copy_b, tBgB(_,_,_,k_next), tBsB(_,_,_,p));
    cp_async_fence(); --k_count; if (k_count > 0) ++k_next;
  }
  int pipe_r = 0, pipe_w = K_PIPE - 1;
  auto tXsA_p = tXsA(_,_,_,pipe_r); auto tXsB_p = tXsB(_,_,_,pipe_r);
  if (K_BLK > 1) {
    cp_async_wait<K_PIPE - 2>(); __syncthreads();
    copy(S2RA{}, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(S2RB{}, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
  }
  CUTE_NO_UNROLL
  while (k_count > -(K_PIPE - 1)) {
    CUTE_UNROLL
    for (int kb = 0; kb < K_BLK; ++kb) {
      if (kb == K_BLK - 1) {
        tXsA_p = tXsA(_,_,_,pipe_r); tXsB_p = tXsB(_,_,_,pipe_r);
        cp_async_wait<K_PIPE - 2>(); __syncthreads();
      }
      auto kn = (kb + Int<1>{}) % K_BLK;
      copy(S2RA{}, tXsA_p(_,_,kn), tXrA(_,_,kn));
      copy(S2RB{}, tXsB_p(_,_,kn), tXrB(_,_,kn));
      if (kb == 0) {
        copy(copy_a, tAgA(_,_,_,k_next), tAsA(_,_,_,pipe_w));
        copy(copy_b, tBgB(_,_,_,k_next), tBsB(_,_,_,pipe_w));
        cp_async_fence(); --k_count; if (k_count > 0) ++k_next;
        pipe_w = pipe_r; pipe_r = (pipe_r == K_PIPE - 1) ? 0 : pipe_r + 1;
      }
      gemm(mma, tCrA(_,_,kb), tCrB(_,_,kb), tCrC);
    }
  }

  // ── Epilogue ──
  int trs = i_tile * TILE_M, tcs = j_tile * TILE_M;
  bool is_diagonal = (i_tile == j_tile);
  auto cC = make_identity_tensor(make_shape(Int<TILE_M>{}, Int<TILE_M>{}));
  auto tCcC = thr_mma.partition_C(cC);

  Tensor out = make_tensor_like<TC>(tCrC);

  if constexpr (EpilogueMode == 0) {
    // GEMM1: D = bf16(acc)
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) out(i) = TC(tCrC(i));
  } else {
    // GEMM2: D = bf16(c*acc + b*src + a*(i==j))
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
      int lr = int(get<0>(tCcC(i))), lc = int(get<1>(tCcC(i)));
      int gr = trs + lr, gc = tcs + lc;
      float sv = float(src_ptr[gr * M + gc]);
      float r = NS_C * tCrC(i) + NS_B * sv;
      if (gr == gc) r += NS_A;
      out(i) = TC(r);
    }
  }

  // Vectorized write to lower triangle position
  copy(out, tCgC);

  // Dual-write mirror to upper triangle
  if (is_diagonal) {
    CUTE_UNROLL
    for (int i = 0; i < size(out); ++i) {
      int gr = trs + int(get<0>(tCcC(i))), gc = tcs + int(get<1>(tCcC(i)));
      if (gr > gc) C_ptr[gc * M + gr] = out(i);
    }
  } else {
    __syncthreads();
    uint16_t* smem_tile = reinterpret_cast<uint16_t*>(smem);
    constexpr int PADDED_N = TILE_M + 8;
    CUTE_UNROLL
    for (int i = 0; i < size(out); ++i) {
      int lr = int(get<0>(tCcC(i))), lc = int(get<1>(tCcC(i)));
      smem_tile[lr * PADDED_N + lc] = reinterpret_cast<const uint16_t&>(out(i));
    }
    __syncthreads();
    int tid = threadIdx.x;
    for (int mr = 0; mr < TILE_M; ++mr) {
      if (tid < TILE_M) {
        int gr = tcs + mr, gc = trs + tid;
        reinterpret_cast<uint16_t*>(C_ptr)[gr * M + gc] = smem_tile[tid * PADDED_N + mr];
      }
    }
  }
}

}  // namespace detail
}  // namespace xcompute
