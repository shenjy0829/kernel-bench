import torch
import triton
import triton.language as tl

@triton.jit
def reduction_kernel(
    input_ptr, temp_output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    
    pid = tl.program_id(0)
    
    input_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offsets < N 

    input = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
    output = tl.sum(input, axis=0)

    tl.store(temp_output_ptr + pid, output)

def solve(input: int, output: int, N: int):
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    temp_output = torch.empty(grid[0], device=input.device, dtype=input.dtype)

    reduction_kernel[grid](
        input, temp_output, N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    final_sum = torch.sum(temp_output)
    output.copy_(final_sum)


# each block process a chunk of data and computes partial sum
# then final sum is computed on CPU