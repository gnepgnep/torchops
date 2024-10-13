import torch
import numpy as np
from torch import Tensor
import typing

__all__ = ["mymul", "argmax", "layernorm_welford", "layernorm"]


def mymul(a: Tensor, b: Tensor) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.cuda_ext.mymul.default(a, b)


def argmax(
        matrix: typing.Union[torch.cuda.FloatTensor, torch.cuda.HalfTensor], 
        dim: int, 
        keepdim: bool=False) -> torch.cuda.CharTensor:
    
    assert matrix.dtype in [torch.float32, torch.float16], "only support float32 and float16"
    assert isinstance(matrix, torch.Tensor) and matrix.is_cuda, "only support torch.Tensor, gpu"
    assert matrix.ndim > dim, f"matrix.ndim({matrix.ndim}) should be larger than dim({dim})"
    assert dim >= 0, f"dim({dim}) should be >= 0"

    shape = [i for i in matrix.shape]
    assert shape[dim] <= 128, f"shape[dim]({shape[dim]}) should be <= 128"

    number, stride, length = 1, 1, shape[dim]
    
    for i in range(dim+1, matrix.ndim):
        stride *= shape[i]

    number = stride

    for i in range(dim):
        number *= shape[i]

    if keepdim:
        shape[dim] = 1
    else:
        shape.pop(dim)

    index_cuda = torch.zeros(shape, dtype = torch.int8, device=matrix.device)

    torch.ops.cuda_ext.argmax(matrix, index_cuda, number, stride, length)

    return index_cuda



def layernorm_welford(
        matrix: typing.Union[torch.cuda.FloatTensor, torch.cuda.HalfTensor]
        ) -> typing.Union[torch.cuda.FloatTensor, torch.cuda.HalfTensor]:
    
    assert matrix.dtype in [torch.float32, torch.float16], "only support float32 and float16"
    assert isinstance(matrix, torch.Tensor) and matrix.is_cuda, "only support torch.Tensor, gpu"
    assert len(matrix.shape) >= 2, "matrix.shape should be >= 2"

    shape = [i for i in matrix.shape]

    M, N = np.prod(shape[0:-1], dtype=np.int32), shape[-1]

    result_cuda = torch.zeros(shape, dtype = matrix.dtype, device=matrix.device)

    torch.ops.cuda_ext.layernorm_welford(matrix, result_cuda, M, N)

    return result_cuda


def layernorm(
        matrix: typing.Union[torch.cuda.FloatTensor, torch.cuda.HalfTensor]
        ) -> typing.Union[torch.cuda.FloatTensor, torch.cuda.HalfTensor]:
    
    assert matrix.dtype in [torch.float32, torch.float16], "only support float32 and float16"
    assert isinstance(matrix, torch.Tensor) and matrix.is_cuda, "only support torch.Tensor, gpu"
    assert len(matrix.shape) >= 2, "matrix.shape should be >= 2"

    shape = [i for i in matrix.shape]

    M, N = np.prod(shape[0:-1], dtype=np.int32), shape[-1]

    result_cuda = torch.zeros(shape, dtype = matrix.dtype, device=matrix.device)

    torch.ops.cuda_ext.layernorm(matrix, result_cuda, M, N)

    return result_cuda