#include <torch/extension.h>
#include <stdio.h>
#include <ATen/ATen.h>
#include <algorithm>
#include "utils_cuda.cuh"

using namespace ::at;

namespace cuda_ext {

namespace {

const int kWarpSize = 32;

template<typename T>
__device__ __inline__ void WelfordCombine(T b_mean, T b_m2, T b_count,
                                          T* mean, T* m2, T* count) {
  if (b_count == 0) {
    return;
  }
  T new_count = *count + b_count;
  T delta = b_mean - *mean;
  T new_mean = *mean + delta * (b_count / new_count);
  T delta2 = b_mean - new_mean;
  T new_m2 = *m2 + b_m2 + delta * delta2 * (*count);

  *mean = new_mean;
  *m2 = new_m2;
  *count = new_count;
}

template<typename T, int thread_group_width = kWarpSize>
__device__ __inline__  void WelfordWarpReduce(T thread_mean, T thread_m2, T thread_count, T* mean,
                                             T* m2, T* count) {
  *mean = thread_mean;
  *m2 = thread_m2;
  *count = thread_count;
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    T b_mean = __shfl_down_sync(0xffffffff, *mean, mask);
    T b_m2 = __shfl_down_sync(0xffffffff, *m2, mask);
    T b_count = __shfl_down_sync(0xffffffff, *count, mask);
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }
}

template<typename T, int thread_group_width = kWarpSize>
__device__ __inline__  void WelfordWarpAllReduce(T thread_mean, T thread_m2, T thread_count, T* mean,
                                                T* m2, T* count, T* shared_mean, T* shared_m2, T* shared_count) {
  
  WelfordWarpReduce<T, thread_group_width>(thread_mean, thread_m2, thread_count, mean, m2, count);
  
  int tid = threadIdx.x;
  int wid = threadIdx.x >> 5;
  int lane_id = tid & 0x1f;
  shared_mean[wid] = lane_id == 0 ? *mean : 0.0f;
  shared_m2[wid] = lane_id == 0 ? *m2 : 0.0f;
  shared_count[wid] = lane_id == 0 ? *count : 0.0f;
  __syncthreads();

  if (wid == 0) {
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    T b_mean = (lane_id < num_warps) ? shared_mean[lane_id] : 0.0f;
    T b_m2 = (lane_id < num_warps) ? shared_m2[lane_id] : 0.0f;
    T b_count = (lane_id < num_warps) ? shared_count[lane_id] : 0.0f;
    WelfordWarpReduce<T, thread_group_width>(b_mean, b_m2, b_count, mean, m2, count);
    if (lane_id == 0) {
        shared_mean[0] = *mean;
        shared_m2[0] = *m2;
        shared_count[0] = *count;
    }
  }
  __syncthreads();
}

template <typename T>
__global__ void layernorm_welford_kernel(void* data, void* result, int64_t N) {
    const T* input = reinterpret_cast<const T*>(data);
    T* output = reinterpret_cast<T*>(result);
    
    __shared__ T shared_mean[32];  // Assuming max 32 warps per block
    __shared__ T shared_m2[32];
    __shared__ T shared_count[32];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    T thread_mean = 0;
    T thread_m2 = 0;
    T thread_count = 0;
    
    // Compute local sum for each thread
    for (int i = tid; i < N; i += blockDim.x) {
        T val = input[bid * N + i];
        thread_count += 1;
        T delta = val - thread_mean;
        thread_mean += delta / thread_count;
        T delta2 = val - thread_mean;
        thread_m2 += delta * delta2;
    }

    // Perform Welford's algorithm reduction within each warp
    T mean, m2, count;
    WelfordWarpAllReduce<T>(thread_mean, thread_m2, thread_count, &mean, &m2, &count, shared_mean, shared_m2, shared_count);

    mean = shared_mean[0];
    m2 = shared_m2[0];
    count = shared_count[0];
    __syncthreads();
    // Compute variance and standard deviation
    T variance = m2 / count;
    T stddev = sqrt(variance + 1e-5);  // Add epsilon for numerical stability
    // Normalize the input
    for (int i = tid; i < N; i += blockDim.x) {
        T val = input[bid * N + i];
        output[bid * N + i] = (val - mean) / stddev;
    }    
}

void layernorm_welford_cuda(Tensor data, Tensor result, int64_t M, int64_t N) {
    data = data.contiguous();
    result = result.contiguous();
    int gpu_id = data.device().index();
    cudaSetDevice(gpu_id);

    int block_size = std::min(static_cast<int>(cdiv(N, 32) * 32), 1024);
    layernorm_welford_kernel<float><<<M, block_size>>>((void*)data.data_ptr(), (void*)result.data_ptr(), N);
    cudaDeviceSynchronize();
}

}

// Implements the function for a specific backend
TORCH_LIBRARY_IMPL(cuda_ext, CUDA, m) {
    m.impl("cuda_ext::layernorm_welford", TORCH_FN(layernorm_welford_cuda));
}

TORCH_LIBRARY_FRAGMENT(cuda_ext, m) {
    m.def("layernorm_welford(Tensor data, Tensor result, int M, int N) -> ()");
}

}