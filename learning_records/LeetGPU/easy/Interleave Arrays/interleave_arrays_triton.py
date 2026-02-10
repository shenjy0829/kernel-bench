import torch
import triton
import triton.language as tl


@triton.jit
def interleave_kernel(
    A_ptr, B_ptr, output_ptr, 
    N, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)       # [0, 1, 2, ..., BLOCK_SIZE-1] + pid * BLOCK_SIZE
    mask = offsets < N
    A_vals = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    B_vals = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    interleaved_offsets = offsets * 2                           # [0, 2, 4, ..., 2*(BLOCK_SIZE-1)] + pid * BLOCK_SIZE * 2
    tl.store(output_ptr + interleaved_offsets, A_vals, mask=mask)
    tl.store(output_ptr + interleaved_offsets + 1, B_vals, mask=mask)    


# A, B, output are tensors on the GPU
# [A[0], B[0], A[1], B[1], A[2], B[2], ...]
def solve(A: torch.Tensor, B: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    interleave_kernel[grid](
        A, B, output, 
        N, 
        BLOCK_SIZE=BLOCK_SIZE)
