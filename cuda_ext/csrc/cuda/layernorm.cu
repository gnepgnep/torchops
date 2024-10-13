#include <torch/extension.h>
#include <stdio.h>
#include <ATen/ATen.h>
#include <algorithm>
#include "utils_cuda.cuh"

using namespace ::at;

namespace cuda_ext {

namespace {

__device__ __forceinline__ float WarpReduceSum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16, 32);
    val += __shfl_xor_sync(0xffffffff, val, 8, 32);
    val += __shfl_xor_sync(0xffffffff, val, 4, 32);
    val += __shfl_xor_sync(0xffffffff, val, 2, 32);
    val += __shfl_xor_sync(0xffffffff, val, 1, 32);
    return val;
}

__device__ __forceinline__ float BlockReduceSum(float val, float* buf) {
    int tid = threadIdx.x;
    val = WarpReduceSum(val);

    int wid = tid >> 5;
    int lane = tid & 0x1f;
    if (lane == 0) buf[wid] = val;
    __syncthreads();

    if (wid == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        val = (lane < num_warps) ? buf[lane] : 0;
        val = WarpReduceSum(val);
        if (lane == 0) buf[0] = val;
    }
    __syncthreads();
    return buf[0];
}

template <typename T>
__global__ void layernorm_kernel(void* data, void* result, int64_t N) {
    const T* input = reinterpret_cast<const T*>(data);
    T* output = reinterpret_cast<T*>(result);
    
    __shared__ T buf[32];  // Assuming max 32 warps per block
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += (float)input[bid * N + i];
    }
    auto mean = BlockReduceSum(sum, buf) / N;

    sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += (input[bid * N + i] - mean) * (input[bid * N + i] - mean);
    }
    auto var = BlockReduceSum(sum, buf) / N;

    for (int i = tid; i < N; i += blockDim.x) {
        output[bid * N + i] = (input[bid * N + i] - mean) / sqrt(var + 1e-5);
    }
}

void layernorm_cuda(Tensor data, Tensor result, int64_t M, int64_t N) {
    data = data.contiguous();
    result = result.contiguous();
    int gpu_id = data.device().index();
    cudaSetDevice(gpu_id);

    int block_size = std::min(static_cast<int>(cdiv(N, 32) * 32), 1024);
    layernorm_kernel<float><<<M, block_size>>>((void*)data.data_ptr(), (void*)result.data_ptr(), N);
    cudaDeviceSynchronize();
}

}

// Implements the function for a specific backend
TORCH_LIBRARY_IMPL(cuda_ext, CUDA, m) {
    m.impl("cuda_ext::layernorm", TORCH_FN(layernorm_cuda));
}

TORCH_LIBRARY_FRAGMENT(cuda_ext, m) {
    m.def("layernorm(Tensor data, Tensor result, int M, int N) -> ()");
}

}