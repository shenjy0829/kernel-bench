import torch
import triton
import triton.language as tl


# Q, K, V, output are tensors on the GPU
# Q K V -- M x d
# Q @ (K.T @ V)
# 
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    
