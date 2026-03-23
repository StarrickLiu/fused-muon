/**
 * Newton-Schulz step dispatcher.
 * GEMM1 + GEMM2: CuTe SYRK (tile dispatch by m)
 * GEMM3: cuBLAS GemmEx
 */
#include "ns_gemm.h"
#include <cstdio>

namespace xcompute {

// Determine optimal split-K factor for GEMM1 based on n/m ratio
static int choose_splitk(int m, int n) {
  if (n < m * 5) return 1;          // n/m < 5: non-split-K SYRK is better
  // For extreme n/m (>8), cuBLAS fallback is used, split-K not needed
  if (n > m * 8) return 1;
  if (m % 128 != 0) return 1;       // need 128-tile alignment for split-K
  if (m < 1280) return 1;           // too few tiles for 128-tile, overhead dominates

  int Mt = (m + 127) / 128;
  int num_tiles = Mt * (Mt + 1) / 2;
  int sm_capacity = 108 * 2;  // A800: 108 SMs × 2 blocks/SM

  // Target: total blocks (tiles * split_k) should fill ~2 waves
  int target_blocks = sm_capacity * 2;
  int split_k = (target_blocks + num_tiles - 1) / num_tiles;

  // Cap to limit workspace overhead
  if (split_k > 8) split_k = 8;

  // Ensure each split has enough K-iterations (≥ 64 tiles for good pipeline)
  int k_tiles = n / 32;
  while (split_k > 1 && k_tiles / split_k < 64) split_k--;

  return (split_k < 2) ? 1 : split_k;
}

size_t newton_schulz_workspace_size(int m, int n) {
  // A (m×m bf16) + B (m×m bf16), each 256-byte aligned
  size_t per_mat = ((size_t)m * m * 2 + 255) & ~255ULL;
  size_t base = per_mat * 2;

  // Split-K workspace (if needed)
  int sk = choose_splitk(m, n);
  size_t sk_ws = (sk > 1) ? detail::gemm1_splitk_workspace_size(m, sk) : 0;
  sk_ws = (sk_ws + 255) & ~255ULL;

  return base + sk_ws;
}

size_t newton_schulz_workspace_size(int m) {
  return newton_schulz_workspace_size(m, m);  // conservative
}

void newton_schulz_step(
    const void* X, void* X_new,
    int m, int n,
    void* workspace,
    cublasHandle_t handle,
    cudaStream_t stream) {

  // Workspace layout
  size_t per_mat = ((size_t)m * m * 2 + 255) & ~255ULL;  // bf16 = 2 bytes
  void* A = workspace;
  void* B = static_cast<char*>(workspace) + per_mat;

  // Tile selection for GEMM1: consider n/m ratio for split-K
  // For GEMM2 (K=m), always use standard tile selection
  bool use_128_gemm1 = (m > 2048) || (m % 128 == 0 && m % 64 != 0);
  bool use_128_gemm2 = use_128_gemm1;

  // For n >> m, force 128-tile for GEMM1 to enable split-K (64-tile doesn't support it)
  int split_k = 1;
  if (m % 128 == 0) {
    split_k = choose_splitk(m, n);
    if (split_k > 1) use_128_gemm1 = true;
  }

  // Ensure alignment
  if (use_128_gemm1 && m % 128 != 0) {
    // Fall back to 64-tile, disable split-K
    use_128_gemm1 = false;
    split_k = 1;
  }
  if (!use_128_gemm1 && m % 64 != 0) {
    return;
  }
  // GEMM1 dispatch: cuBLAS fallback for shapes where SYRK can't compete
  // - m < 1280 with large n: too few 128-tiles even with split-K
  // - n/m > 6 with no split-K: cuBLAS has much better block count
  bool cublas_gemm1 = false;
  if (m <= 1280) cublas_gemm1 = true;               // small m: SYRK block count too low
  if (n > m * 8) cublas_gemm1 = true;               // extreme n/m: cuBLAS full GEMM is competitive

  if (cublas_gemm1) {
    // cuBLAS: A = X @ X^T as full GEMM (faster than SYRK for n >> m)
    float g1_one = 1.0f, g1_zero = 0.0f;
    cublasSetStream(handle, stream);
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        m, m, n, &g1_one,
        X, CUDA_R_16BF, n, X, CUDA_R_16BF, n, &g1_zero,
        A, CUDA_R_16BF, m,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  } else if (split_k > 1) {
    void* sk_ws = static_cast<char*>(workspace) + per_mat * 2;
    detail::launch_gemm1_128_splitk(X, A, m, n, sk_ws, split_k, stream);
  } else if (use_128_gemm1) {
    detail::launch_gemm1_128(X, A, m, n, stream);
  } else {
    detail::launch_gemm1_64(X, A, m, n, stream);
  }

  // GEMM2: B = c*(A@A) + b*A + a*I (SYRK, fused epilogue, dual-write)
  // GEMM2 K=m, use standard tile selection (no split-K needed)
  if (use_128_gemm2) {
    detail::launch_gemm2_128(A, B, m, stream);
  } else {
    detail::launch_gemm2_64(A, B, m, stream);
  }

  // GEMM3: X_new = B @ X (cuBLAS)
  // Row-major: C(m,n) = B(m,m) @ X(m,n)
  // cuBLAS col-major: C^T(n,m) = X^T(n,m) @ B^T(m,m)
  // B is symmetric → B^T = B
  // So: cublasGemmEx(OP_N, OP_N, n, m, m, X, B, X_new) with col-major LDs
  float one = 1.0f, zero = 0.0f;
  cublasSetStream(handle, stream);
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      n, m, m,
      &one,
      X, CUDA_R_16BF, n,       // X^T in col-major = X in row-major, ld=n
      B, CUDA_R_16BF, m,       // B^T = B (symmetric), ld=m
      &zero,
      X_new, CUDA_R_16BF, n,   // X_new^T in col-major = X_new in row-major, ld=n
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

}  // namespace xcompute
