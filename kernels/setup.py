from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="paged_attn_cuda",
    ext_modules=[
        CUDAExtension(
            name="paged_attn_cuda",
            sources=["paged_attention.cu"],
            extra_compile_args={"nvcc": ["-O2", "--use_fast_math"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
