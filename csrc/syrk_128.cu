/**
 * SYRK 128×128 tile kernels (GEMM1 + GEMM2).
 * Compiled separately to avoid register pressure cross-contamination with 64×64.
 * Config: K32_S4_CG (BYPASS L1), 4 warps, 232-254 regs, 2 blocks/SM.
 */
#include "syrk_common.cuh"
#include "syrk_splitk.cuh"
#include "ns_gemm.h"

namespace xcompute {
namespace detail {

using namespace cute;

// 128×128 CuTe configuration
static auto make_config_128() {
  auto sw = composition(Swizzle<2,3,3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{});
  auto g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
                              Layout<Shape<_32,_4>, Stride<_4,_1>>{}, Layout<Shape<_1,_8>>{});
  Copy_Atom<SM75_U32x4_LDSM_N, Element> s2r;
  auto mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                             Layout<Shape<_2,_2>>{}, Tile<_32,_32,_16>{});
  return cute::make_tuple(sw, g2s, s2r, mma);
}

template <int EpilogueMode>
static void launch_syrk_128(const void* input, void* output, int m, int k,
                              const void* src, cudaStream_t stream) {
  auto [sw, g2s, s2r, mma] = make_config_128();

  auto prob = make_shape(m, m, k);
  auto dA = make_stride(k, Int<1>{});
  auto dC = make_stride(m, Int<1>{});
  auto cta = make_shape(Int<128>{}, Int<128>{}, Int<32>{});
  auto sA = tile_to_shape(sw, make_shape(Int<128>{}, Int<32>{}, Int<4>{}));
  auto sB = tile_to_shape(sw, make_shape(Int<128>{}, Int<32>{}, Int<4>{}));
  auto sC = make_layout(make_shape(Int<128>{}, Int<128>{}));

  int Mt = (m + 127) / 128;
  int smem_sz = int(sizeof(SmemStorage<Element, Element, decltype(sA), decltype(sB)>));

  auto kern = syrk_kernel<EpilogueMode, 128,
      decltype(prob), decltype(cta),
      Element, decltype(dA), decltype(sA), decltype(g2s), decltype(s2r),
      Element, decltype(dA), decltype(sB), decltype(g2s), decltype(s2r),
      Element, decltype(dC), decltype(sC), decltype(mma)>;

  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);
  cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  dim3 grid(Mt * (Mt + 1) / 2), block(int(size(mma)));
  kern<<<grid, block, smem_sz, stream>>>(prob, cta,
      static_cast<const Element*>(input), dA, sA, g2s, s2r,
      static_cast<const Element*>(input), dA, sB, g2s, s2r,
      static_cast<Element*>(output), dC, sC, mma, Mt,
      static_cast<const Element*>(src));
}

void launch_gemm1_128(const void* X, void* A, int m, int n, cudaStream_t stream) {
  launch_syrk_128<0>(X, A, m, n, nullptr, stream);
}

void launch_gemm2_128(const void* A, void* B, int m, cudaStream_t stream) {
  launch_syrk_128<1>(A, B, m, m, A, stream);
}

// Split-K GEMM1: for n >> m
size_t gemm1_splitk_workspace_size(int m, int split_k) {
  int Mt = (m + 127) / 128;
  int num_tiles = Mt * (Mt + 1) / 2;
  int threads = 128;  // 4 warps
  int elems_per_thread = 128;  // 128×128 / 128 threads
  return (size_t)num_tiles * split_k * threads * elems_per_thread * sizeof(float);
}

void launch_gemm1_128_splitk(const void* X, void* A, int m, int n,
                              void* splitk_workspace, int split_k, cudaStream_t stream) {
  auto [sw, g2s, s2r, mma] = make_config_128();

  auto prob = make_shape(m, m, n);
  auto dA = make_stride(n, Int<1>{});
  auto dC = make_stride(m, Int<1>{});
  auto cta = make_shape(Int<128>{}, Int<128>{}, Int<32>{});
  auto sA = tile_to_shape(sw, make_shape(Int<128>{}, Int<32>{}, Int<4>{}));
  auto sB = tile_to_shape(sw, make_shape(Int<128>{}, Int<32>{}, Int<4>{}));
  auto sC = make_layout(make_shape(Int<128>{}, Int<128>{}));

  int Mt = (m + 127) / 128;
  int num_tiles = Mt * (Mt + 1) / 2;
  int smem_sz = int(sizeof(SmemStorage<Element, Element, decltype(sA), decltype(sB)>));

  // Stage 1: partial accumulation
  auto kern1 = syrk_splitk_partial<128,
      decltype(prob), decltype(cta),
      Element, decltype(dA), decltype(sA), decltype(g2s), decltype(s2r),
      Element, decltype(dA), decltype(sB), decltype(g2s), decltype(s2r),
      Element, decltype(dC), decltype(sC), decltype(mma)>;
  cudaFuncSetAttribute(kern1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);
  cudaFuncSetAttribute(kern1, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  dim3 grid1(num_tiles * split_k), block1(int(size(mma)));
  kern1<<<grid1, block1, smem_sz, stream>>>(prob, cta,
      static_cast<const Element*>(X), dA, sA, g2s, s2r,
      static_cast<const Element*>(X), dA, sB, g2s, s2r,
      static_cast<float*>(splitk_workspace),
      dC, sC, mma, Mt, split_k, num_tiles);

  // Stage 2: reduce + epilogue (GEMM1 mode = 0) + dual-write
  // smem for dual-write mirror: TILE * (TILE+8) * 2 bytes
  int reduce_smem = 128 * (128 + 8) * 2;
  auto kern2 = syrk_splitk_reduce<0, 128,
      decltype(prob), decltype(cta),
      Element, decltype(dC), decltype(sC), decltype(mma)>;
  cudaFuncSetAttribute(kern2, cudaFuncAttributeMaxDynamicSharedMemorySize, reduce_smem);

  dim3 grid2(num_tiles), block2(int(size(mma)));
  kern2<<<grid2, block2, reduce_smem, stream>>>(prob, cta,
      static_cast<const float*>(splitk_workspace),
      static_cast<Element*>(A), dC, sC, mma, Mt,
      nullptr, split_k, num_tiles);
}

}  // namespace detail
}  // namespace xcompute
