from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import subprocess

CUTLASS_PATH = "./cutlass"

if not os.path.exists(CUTLASS_PATH):
    print("CUTLASS not found. Cloning the repository...")
    subprocess.run([
        "git", "clone", "https://github.com/NVIDIA/cutlass.git", CUTLASS_PATH
    ], check=True)
    print("CUTLASS cloned successfully.")

setup(
    name='cutlass_fp8',
    ext_modules=[
        CUDAExtension(
            'cutlass_fp8',
            ['cutlass_fp8.cpp', 'cutlass_fp8_kernel.cu'],
            include_dirs=[
                os.path.join(CUTLASS_PATH, 'include'),
                os.path.join(CUTLASS_PATH, 'tools/util/include'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_89',
                    '--ptxas-options=-v',
                    '-std=c++17',
                    f'-I{CUTLASS_PATH}/include',
                    f'-I{CUTLASS_PATH}/tools/util/include'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

