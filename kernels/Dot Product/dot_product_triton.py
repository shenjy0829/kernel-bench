import torch
import triton
import triton.language as tl

@triton.jit
def dot_product_kernel(
    A_ptr, B_ptr, result_ptr, 
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    input_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offsets < N
    
    input_A = tl.load( A_ptr + input_offsets, mask = input_mask, other=0.0)
    input_B = tl.load( B_ptr + input_offsets, mask = input_mask, other=0.0)

    output = tl.sum(input_A * input_B, axis=0)

    tl.atomic_add(result_ptr, output)


# a, b, result are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    result.zero_()
    dot_product_kernel[grid](
        A, B, result,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
