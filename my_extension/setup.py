from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
import os
import torch

# Set the LD_LIBRARY_PATH
os.environ['LD_LIBRARY_PATH'] = '/root/miniconda3/lib/python3.10/site-packages/torch/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

setup(
    name='my_extension',
    ext_modules=[
        CUDAExtension(
            'my_extension',  # Name of the extension
            sources=['my_extension.cpp', 'my_extension_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-gencode', 'arch=compute_75,code=sm_75']
            },
            include_dirs=[torch.utils.cpp_extension.include_paths()]  # Ensure PyTorch headers are found
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


# from setuptools import setup
# from torch.utils.cpp_extension import CppExtension, BuildExtension
# import torch

# setup(
#     name='my_extension',
#     ext_modules=[
#         CppExtension(
#             'my_extension',
#             ['my_extension.cpp'],
#             extra_compile_args=['-O3'],  # Optimize compiler flags
#             include_dirs=[torch.utils.cpp_extension.include_paths()]  # Ensure PyTorch headers are found
#         )
#     ],
#     cmdclass={'build_ext': BuildExtension}
# )


