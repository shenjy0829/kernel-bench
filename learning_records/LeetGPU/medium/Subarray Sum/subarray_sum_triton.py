import torch
import triton
import triton.language as tl


@triton.jit
def subarray_sum_kernel(
    input_ptr, output_ptr, 
    N, S, E,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    input_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offsets < (E - S + 1)

    input = tl.load(input_ptr + S + input_offsets, mask=input_mask, other=0.0)
    output = tl.sum(input, axis=0)
    
    tl.atomic_add(output_ptr, output)

            
# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, S: int, E: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(E - S + 1, BLOCK_SIZE), )

    output.zero_()
    subarray_sum_kernel[grid](
        input, output, 
        N, S, E, 
        BLOCK_SIZE
    )