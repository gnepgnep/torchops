#include <torch/extension.h>
#include <stdio.h>
#include <ATen/ATen.h>
#include "utils_cuda.cuh"

using namespace ::at;

template <typename T>
__device__ bool greater_function(T data1, T data2) {
    return data1 > data2;
}

namespace cuda_ext {

namespace {

template <typename T>
__device__ int max_index_function(T* data, int stride, int length) {
    T max_value = data[0];
    int max_index = 0;

    int index = 0;
    for (int i=1; i<length; ++i) {
        index += stride;
        if (greater_function<T>(data[index], max_value)) {
            max_value = data[index];
            max_index = i;
        }
    }
    return max_index;
}


template <typename T>
__global__ void argmax_kernel(void* data, char* argmax_index, int number, int stride, int length) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < number) {
        int row = i / stride;
        int col = i % stride;
        argmax_index[i] = max_index_function<T>((T*)data + row * length * stride + col, stride, length);
    }
}

void argmax_cuda(Tensor data, Tensor index, int64_t number, int64_t stride, int64_t length) {
    data = data.contiguous();
    index = index.contiguous();
    int gpu_id = index.device().index();
    cudaSetDevice(gpu_id);

    int block_size = 32;
    argmax_kernel<float><<<cdiv(1.0*number, block_size), block_size>>>((void*)data.data_ptr(), (char*)index.data_ptr(), number, stride, length);
    cudaDeviceSynchronize();
}

}

// Implements the function for a specific backend
TORCH_LIBRARY_IMPL(cuda_ext, CUDA, m) {
    m.impl("cuda_ext::argmax", TORCH_FN(argmax_cuda));
}

TORCH_LIBRARY_FRAGMENT(cuda_ext, m) {
    m.def("argmax(Tensor data, Tensor index, int number, int stride, int length) -> ()");
}

}