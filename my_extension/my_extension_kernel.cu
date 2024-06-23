#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for addition
template <typename scalar_t>
__global__ void add_kernel(const scalar_t* a, const scalar_t* b, scalar_t* result, int64_t size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = a[index] + b[index];
    }
}

void add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor result) {
    const int threads = 1024;
    const int blocks = (a.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "add_cuda", ([&] {
        add_kernel<<<blocks, threads>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            a.numel()
        );
    }));

    // Wait for the kernel to complete
    cudaDeviceSynchronize();
}
