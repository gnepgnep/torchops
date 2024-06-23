import torch
import my_extension

a = torch.randn(1000, device='cuda', dtype=torch.float64)
b = torch.randn(1000, device='cuda', dtype=torch.float64)
result = torch.empty_like(a)

my_extension.add(a, b, result)

print(result)
