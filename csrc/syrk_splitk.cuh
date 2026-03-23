#pragma once
#include "syrk_common.cuh"

namespace xcompute {
namespace detail {

using namespace cute;

// ============================================================
// SYRK Split-K: Two-stage approach
// Stage 1: Each block accumulates partial f32 results into workspace
//          workspace[split_idx][tile_idx][TILE*TILE] in MMA partition order
// Stage 2: Reduction kernel sums partials, applies epilogue, dual-write
//
// Workspace uses f32 to avoid bf16 precision loss during reduction.
// Layout: contiguous per (tile, split) for locality.
// ============================================================

// Stage 1: Partial accumulation kernel
template <int TILE_M,
          class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class CopyA, class S2RA,
          class TB, class BStride, class BSmemLayout, class CopyB, class S2RB,
          class TC, class CStride, class CSmemLayout, class MMA>
__global__ static __launch_bounds__(decltype(size(MMA{}))::value)
void syrk_splitk_partial(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 TA const* A_ptr, AStride dA, ASmemLayout, CopyA copy_a, S2RA,
                 TB const* B_ptr, BStride dB, BSmemLayout, CopyB copy_b, S2RB,
                 float* workspace,
                 CStride dC, CSmemLayout, MMA mma, int M_tiles,
                 int split_k, int num_tiles) {
  int tile_idx = blockIdx.x / split_k;
  int split_idx = blockIdx.x % split_k;

  int i_tile, j_tile;
  linear_to_lower_tri(tile_idx, i_tile, j_tile);
  if (i_tile >= M_tiles) return;

  Tensor mA = make_tensor(make_gmem_ptr(A_ptr), select<0,2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B_ptr), select<1,2>(shape_MNK), dB);
  Tensor gA = local_tile(mA, cta_tiler, make_coord(i_tile, _, _), Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, make_coord(_, j_tile, _), Step< X, _1, _1>{});

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

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  // Need a dummy gC to get correct fragment shape
  int M = get<0>(shape_MNK);
  Tensor mC_dummy = make_tensor(make_gmem_ptr((TC*)nullptr), select<0,1>(shape_MNK), dC);
  Tensor gC_dummy = local_tile(mC_dummy, cta_tiler, make_coord(i_tile, j_tile, _), Step<_1, _1, X>{});
  auto tCgC_dummy = thr_mma.partition_C(gC_dummy);
  auto tCrC = thr_mma.make_fragment_C(tCgC_dummy);
  clear(tCrC);

  auto tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
  auto tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
  auto s2r_a = make_tiled_copy_A(S2RA{}, mma);
  auto s2r_b = make_tiled_copy_B(S2RB{}, mma);
  auto tXsA = s2r_a.get_slice(threadIdx.x).partition_S(sA);
  auto tXrA = s2r_a.get_slice(threadIdx.x).retile_D(tCrA);
  auto tXsB = s2r_b.get_slice(threadIdx.x).partition_S(sB);
  auto tXrB = s2r_b.get_slice(threadIdx.x).retile_D(tCrB);
  auto K_BLK = size<2>(tCrA);

  // K range for this split
  int total_k = size<3>(tAgA);
  int k_per = (total_k + split_k - 1) / split_k;
  int k_start = split_idx * k_per;
  int k_end = min(k_start + k_per, total_k);
  int k_count = k_end - k_start;
  int k_next = k_start;

  if (k_count <= 0) return;

  // Pipeline fill — identical to syrk_common.cuh but with k_count/k_next offset
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
  // Mainloop — identical to syrk_common.cuh
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

  // Write f32 partial to workspace
  // Layout: [num_tiles][split_k][ELEMS][THREADS] — coalesced!
  // Consecutive threads write consecutive addresses in each iteration.
  constexpr int ELEMS = decltype(size(tCrC))::value;
  int tid = threadIdx.x;
  int nthreads = int(size(mma));
  float* ws_base = workspace + ((size_t)tile_idx * split_k + split_idx) * ELEMS * nthreads;
  CUTE_UNROLL
  for (int i = 0; i < ELEMS; ++i)
    ws_base[(size_t)i * nthreads + tid] = tCrC(i);
}

// ============================================================
// Stage 2: Reduction + epilogue + dual-write
// One block per tile. Each thread reduces its own ELEMS across split_k partials,
// then applies epilogue and writes output.
// Must use SAME MMA config to get correct coordinate mapping.
// ============================================================
template <int EpilogueMode, int TILE_M,
          class ProblemShape, class CtaTiler,
          class TC, class CStride, class CSmemLayout, class MMA>
__global__ static __launch_bounds__(decltype(size(MMA{}))::value)
void syrk_splitk_reduce(ProblemShape shape_MNK, CtaTiler cta_tiler,
                 float const* workspace,
                 TC* C_ptr, CStride dC, CSmemLayout, MMA mma, int M_tiles,
                 TC const* src_ptr, int split_k, int num_tiles) {
  int M = get<0>(shape_MNK);
  int tile_idx = blockIdx.x;
  int i_tile, j_tile;
  linear_to_lower_tri(tile_idx, i_tile, j_tile);
  if (i_tile >= M_tiles) return;

  Tensor mC = make_tensor(make_gmem_ptr(C_ptr), select<0,1>(shape_MNK), dC);
  Tensor gC = local_tile(mC, cta_tiler, make_coord(i_tile, j_tile, _), Step<_1, _1, X>{});

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  auto tCgC = thr_mma.partition_C(gC);
  auto tCrC = thr_mma.make_fragment_C(tCgC);

  // Read and sum partials from workspace
  constexpr int ELEMS = decltype(size(tCrC))::value;
  int tid = threadIdx.x;
  int nthreads = int(size(mma));

  // Read and sum partials — coalesced layout [num_tiles][split_k][ELEMS][THREADS]
  clear(tCrC);
  for (int s = 0; s < split_k; ++s) {
    const float* ws_base = workspace + ((size_t)tile_idx * split_k + s) * ELEMS * nthreads;
    CUTE_UNROLL
    for (int i = 0; i < ELEMS; ++i) tCrC(i) += ws_base[(size_t)i * nthreads + tid];
  }

  // Apply epilogue + dual-write (same as non-split-K version)
  int trs = i_tile * TILE_M, tcs = j_tile * TILE_M;
  bool is_diagonal = (i_tile == j_tile);
  auto cC = make_identity_tensor(make_shape(Int<TILE_M>{}, Int<TILE_M>{}));
  auto tCcC = thr_mma.partition_C(cC);

  Tensor out = make_tensor_like<TC>(tCrC);
  if constexpr (EpilogueMode == 0) {
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) out(i) = TC(tCrC(i));
  } else {
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
  copy(out, tCgC);

  // Dual-write mirror
  if (is_diagonal) {
    CUTE_UNROLL
    for (int i = 0; i < size(out); ++i) {
      int gr = trs + int(get<0>(tCcC(i))), gc = tcs + int(get<1>(tCcC(i)));
      if (gr > gc) C_ptr[gc * M + gr] = out(i);
    }
  } else {
    extern __shared__ char smem_reduce[];
    uint16_t* st = reinterpret_cast<uint16_t*>(smem_reduce);
    constexpr int PAD = TILE_M + 8;
    CUTE_UNROLL
    for (int i = 0; i < size(out); ++i) {
      int lr = int(get<0>(tCcC(i))), lc = int(get<1>(tCcC(i)));
      st[lr * PAD + lc] = reinterpret_cast<const uint16_t&>(out(i));
    }
    __syncthreads();
    int t = threadIdx.x;
    for (int mr = 0; mr < TILE_M; ++mr)
      if (t < TILE_M)
        reinterpret_cast<uint16_t*>(C_ptr)[(tcs+mr)*M + trs+t] = st[t*PAD+mr];
  }
}

}  // namespace detail
}  // namespace xcompute
