#include <torch/extension.h>

// Declaration of the CUDA function
void add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor result);

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



