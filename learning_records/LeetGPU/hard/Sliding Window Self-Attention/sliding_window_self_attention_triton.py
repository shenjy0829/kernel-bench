import torch
import triton
import triton.language as tl

@triton.jit
def sliding_window_attention_kernel(


# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d: int,
    window_size: int,
):
    BLOCKSIZE_M = 32
    BLOCKSIZE_d = 32