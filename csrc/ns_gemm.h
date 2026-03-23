#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>

namespace xcompute {

/// Newton-Schulz iteration step:
///   X_new = (c*(X@X^T)^2 + b*(X@X^T) + a*I) @ X
///
/// Coefficients: (a, b, c) = (3.4445, -4.7750, 2.0315)
///
/// @param X         Input matrix, bf16, row-major [m, n]
/// @param X_new     Output matrix, bf16, row-major [m, n]
/// @param m         Number of rows (must be multiple of 128 for 128-tile, 64 for 64-tile)
/// @param n         Number of columns (must be multiple of 32)
/// @param workspace Device memory >= newton_schulz_workspace_size(m) bytes
/// @param handle    cuBLAS handle (for GEMM3)
/// @param stream    CUDA stream
void newton_schulz_step(
    const void* X, void* X_new,
    int m, int n,
    void* workspace,
    cublasHandle_t handle,
    cudaStream_t stream = 0);

/// Returns required workspace size in bytes.
/// Layout: A (m×m bf16) + B (m×m bf16) + optional split-K workspace.
size_t newton_schulz_workspace_size(int m, int n);
size_t newton_schulz_workspace_size(int m);  // conservative (assumes n=m)

// Internal launcher declarations (defined in syrk_128.cu / syrk_64.cu)
namespace detail {

void launch_gemm1_128(const void* X, void* A, int m, int n, cudaStream_t stream);
void launch_gemm2_128(const void* A, void* B, int m, cudaStream_t stream);
void launch_gemm1_64(const void* X, void* A, int m, int n, cudaStream_t stream);
void launch_gemm2_64(const void* A, void* B, int m, cudaStream_t stream);

// Split-K versions for GEMM1 when n >> m
// workspace: f32, size = split_k * num_tiles * threads * elems_per_thread * sizeof(float)
void launch_gemm1_128_splitk(const void* X, void* A, int m, int n,
                              void* splitk_workspace, int split_k, cudaStream_t stream);

// Returns required split-K workspace size in bytes
size_t gemm1_splitk_workspace_size(int m, int split_k);

}  // namespace detail
}  // namespace xcompute
