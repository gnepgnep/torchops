from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_extension',
    ext_modules=[
        CUDAExtension(
            name='my_extension',
            sources=['my_extension.cpp', 'my_extension_kernel.cu'],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
