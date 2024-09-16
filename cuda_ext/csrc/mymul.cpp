#include <torch/extension.h>

#include <vector>

namespace cuda_ext {


at::Tensor mymul_cpu(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i];
  }
  return result;
}



//Defines the operators
TORCH_LIBRARY_FRAGMENT(cuda_ext, m) {
  m.def("mymul(Tensor a, Tensor b) -> Tensor");
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(cuda_ext, CPU, m) {
  m.impl("mymul", &mymul_cpu);
}

}