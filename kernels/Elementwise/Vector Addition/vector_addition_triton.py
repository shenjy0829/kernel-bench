import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE) + BLOCK_SIZE * pid
    mask = offsets < n_elements
    x = tl.load(a + offsets, mask=mask)
    y = tl.load(b + offsets, mask=mask)
    z = x + y
    tl.store( c + offsets, z, mask=mask)
   
# a, b, c are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](A, B, C, N, BLOCK_SIZE)