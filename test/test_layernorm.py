import torch
import torch.nn as nn
import cuda_ext

sentence_length, embedding_dim = 64, 64
embedding = torch.randn(sentence_length, embedding_dim).cuda()
layer_norm = nn.LayerNorm(embedding_dim).cuda()

torch_result = layer_norm(embedding)

cuda_result_welford = cuda_ext.layernorm_welford(embedding)

cuda_result = cuda_ext.layernorm(embedding)

print(torch_result[0][:10])
print(cuda_result[0][:10])
print(cuda_result_welford[0][:10])
