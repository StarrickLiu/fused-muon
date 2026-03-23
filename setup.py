from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
cutlass_include = os.path.join(this_dir, "third_party", "cutlass", "include")

setup(
    name="fused-muon",
    version="0.1.0",
    description="Fused CUDA Muon optimizer with optimized Newton-Schulz iteration",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="muon_fused._fused_muon_C",
            sources=[
                "csrc/torch_binding.cu",
                "csrc/ns_gemm.cu",
                "csrc/syrk_128.cu",
                "csrc/syrk_64.cu",
            ],
            include_dirs=[
                os.path.join(this_dir, "csrc"),
                cutlass_include,
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3", "-std=c++17",
                    "-gencode=arch=compute_80,code=sm_80",
                    "--use_fast_math",
                ],
            },
            libraries=["cublas"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
)
