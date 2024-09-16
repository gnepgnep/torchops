
import torch
import cuda_ext

a = torch.randn(100, device='cuda', dtype=torch.float32)
b = torch.randn(100, device='cuda', dtype=torch.float32)

result = torch.empty_like(a)

result = cuda_ext.mymul(a, b)
print(result)
print()

result = torch.ops.cuda_ext.mymul(a, b)

print(result)

m = torch.rand(32,64,128,128).cuda()

result = cuda_ext.argmax(m, dim=0)
print(result)
# print()

result = cuda_ext.add(a, b)
print(result)
print()
