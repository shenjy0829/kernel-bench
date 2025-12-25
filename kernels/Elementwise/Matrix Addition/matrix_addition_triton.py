import torch
import triton
import triton.language as tl

@triton.jit
def matrix_add_kernel(A, B, C, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_block_1 = tl.load(A + offsets, mask=mask)
    input_block_2 = tl.load(B + offsets, mask=mask)
    output_block = input_block_1 + input_block_2
    tl.store(C + offsets, output_block, mask=mask)
   
# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):    
    BLOCK_SIZE = 1024
    n_elements = N * N
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    matrix_add_kernel[grid](A, B, C, n_elements, BLOCK_SIZE)