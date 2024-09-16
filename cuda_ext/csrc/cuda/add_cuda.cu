#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda_ext {

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

void add(torch::Tensor a, torch::Tensor b, torch::Tensor result) {
    // Check the inputs and outputs for the correct device and type
    TORCH_CHECK(a.device().is_cuda(), "Input tensor 'a' must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor 'b' must be a CUDA tensor");
    TORCH_CHECK(result.device().is_cuda(), "Output tensor must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
    TORCH_CHECK(a.sizes() == result.sizes(), "Output tensor must have the same size as input tensors");

    // Call the CUDA function
    add_cuda(a, b, result);
}

// Define the module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Add two tensors together (CUDA)");
}

}