/**
 * PyTorch C++ extension binding for fused Newton-Schulz iteration.
 * Bridges torch::Tensor ↔ raw CUDA API (ns_gemm.h).
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDAGuard.h>
#include "ns_gemm.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_BF16(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, #x " must be bf16")

// Query workspace size for a given shape
int64_t query_workspace_size(int64_t m, int64_t n) {
  return static_cast<int64_t>(xcompute::newton_schulz_workspace_size(m, n));
}

// Single NS iteration step: X_new = (aI + bA + cA^2) @ X
// where A = X @ X^T
torch::Tensor ns_step(torch::Tensor X, c10::optional<torch::Tensor> workspace) {
  CHECK_CUDA(X); CHECK_CONTIGUOUS(X); CHECK_BF16(X);
  TORCH_CHECK(X.dim() == 2, "X must be 2D, got ", X.dim(), "D");

  int m = X.size(0), n = X.size(1);
  TORCH_CHECK(m % 64 == 0 || m % 128 == 0,
              "m=", m, " must be multiple of 64 or 128");

  auto X_new = torch::empty_like(X);

  // Workspace
  size_t ws_size = xcompute::newton_schulz_workspace_size(m, n);
  torch::Tensor ws_tensor;
  if (workspace.has_value()) {
    ws_tensor = workspace.value();
    TORCH_CHECK(ws_tensor.nbytes() >= ws_size,
                "workspace too small: need ", ws_size, " bytes, got ", ws_tensor.nbytes());
  } else {
    ws_tensor = torch::empty({static_cast<int64_t>(ws_size)},
                             X.options().dtype(torch::kByte));
  }

  // Get cuBLAS handle and stream
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  xcompute::newton_schulz_step(
      X.data_ptr(), X_new.data_ptr(),
      m, n, ws_tensor.data_ptr(),
      handle, stream);

  return X_new;
}

// Full NS iteration: normalize, transpose if needed, run steps, transpose back
// Drop-in replacement for zeropower_via_newtonschulz5()
torch::Tensor fused_newton_schulz(torch::Tensor G, int64_t steps,
                                   c10::optional<torch::Tensor> workspace) {
  CHECK_CUDA(G);
  TORCH_CHECK(G.dim() == 2, "G must be 2D");

  // Cast to bf16
  auto X = G.to(torch::kBFloat16).contiguous();

  // Transpose if m > n (work with the smaller dimension as m)
  bool transposed = false;
  if (X.size(0) > X.size(1)) {
    X = X.t().contiguous();
    transposed = true;
  }

  int m = X.size(0), n = X.size(1);

  // Check alignment - fall back to PyTorch if not aligned
  bool can_use_fused = (m % 64 == 0) && (n % 32 == 0);
  if (!can_use_fused) {
    // Fallback: pure PyTorch NS iteration
    float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    X = X / (X.norm() + 1e-7f);
    for (int64_t i = 0; i < steps; ++i) {
      auto A = X.mm(X.t());
      auto B = b * A + c * A.mm(A);
      X = a * X + B.mm(X);
    }
    if (transposed) X = X.t().contiguous();
    return X;
  }

  // Normalize
  X = X / (X.norm() + 1e-7f);
  X = X.contiguous();

  // Allocate workspace
  size_t ws_size = xcompute::newton_schulz_workspace_size(m, n);
  torch::Tensor ws_tensor;
  if (workspace.has_value()) {
    ws_tensor = workspace.value();
    if (static_cast<size_t>(ws_tensor.nbytes()) < ws_size) {
      ws_tensor = torch::empty({static_cast<int64_t>(ws_size)},
                               X.options().dtype(torch::kByte));
    }
  } else {
    ws_tensor = torch::empty({static_cast<int64_t>(ws_size)},
                             X.options().dtype(torch::kByte));
  }

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Ping-pong between two buffers
  auto X_new = torch::empty_like(X);
  for (int64_t i = 0; i < steps; ++i) {
    xcompute::newton_schulz_step(
        X.data_ptr(), X_new.data_ptr(),
        m, n, ws_tensor.data_ptr(),
        handle, stream);
    std::swap(X, X_new);
  }

  if (transposed) X = X.t().contiguous();
  return X;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("newton_schulz_step", &ns_step,
        "Single Newton-Schulz step (X_new = f(X))",
        py::arg("X"), py::arg("workspace") = py::none());
  m.def("fused_newton_schulz", &fused_newton_schulz,
        "Full Newton-Schulz iteration (drop-in replacement for zeropower_via_newtonschulz5)",
        py::arg("G"), py::arg("steps") = 5, py::arg("workspace") = py::none());
  m.def("workspace_size", &query_workspace_size,
        "Query workspace size in bytes for given (m, n)",
        py::arg("m"), py::arg("n"));
}
