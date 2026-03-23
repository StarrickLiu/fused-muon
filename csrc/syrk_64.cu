/**
 * SYRK 64×64 tile kernels (GEMM1 + GEMM2).
 * Compiled separately to avoid register pressure cross-contamination with 128×128.
 * Config: K32_S3_CG, 4 warps, ~90 regs, 5 blocks/SM.
 */
#include "syrk_common.cuh"
#include "ns_gemm.h"

namespace xcompute {
namespace detail {

using namespace cute;

template <int EpilogueMode>
static void launch_syrk_64(const void* input, void* output, int m, int k,
                             const void* src, cudaStream_t stream) {
  auto sw = composition(Swizzle<2,3,3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{});
  auto g2s = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{},
                              Layout<Shape<_32,_4>, Stride<_4,_1>>{}, Layout<Shape<_1,_8>>{});
  Copy_Atom<SM75_U32x4_LDSM_N, Element> s2r;
  auto mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                             Layout<Shape<_2,_2>>{}, Tile<_32,_32,_16>{});

  auto prob = make_shape(m, m, k);
  auto dA = make_stride(k, Int<1>{});
  auto dC = make_stride(m, Int<1>{});
  auto cta = make_shape(Int<64>{}, Int<64>{}, Int<32>{});
  auto sA = tile_to_shape(sw, make_shape(Int<64>{}, Int<32>{}, Int<3>{}));
  auto sB = tile_to_shape(sw, make_shape(Int<64>{}, Int<32>{}, Int<3>{}));
  auto sC = make_layout(make_shape(Int<64>{}, Int<64>{}));

  int Mt = (m + 63) / 64;
  int smem_sz = int(sizeof(SmemStorage<Element, Element, decltype(sA), decltype(sB)>));

  auto kern = syrk_kernel<EpilogueMode, 64,
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

void launch_gemm1_64(const void* X, void* A, int m, int n, cudaStream_t stream) {
  launch_syrk_64<0>(X, A, m, n, nullptr, stream);
}

void launch_gemm2_64(const void* A, void* B, int m, cudaStream_t stream) {
  launch_syrk_64<1>(A, B, m, m, A, stream);
}

}  // namespace detail
}  // namespace xcompute
