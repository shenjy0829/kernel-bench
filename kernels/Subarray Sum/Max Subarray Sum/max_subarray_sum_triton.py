import torch
import triton
import triton.language as tl

@triton.jit
def max_subarray_kernel(
    input_ptr, block_output_ptr, 
    N, window_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    
    current_sum = 0.0
    elements_loaded = 0

    while elements_loaded < window_size:
        chunk_start = elements_loaded
        chunk_end = tl.minimum(BLOCK_SIZE, window_size - elements_loaded)

        mask = offset < chunk_end

        chunk_data = tl.load(input_ptr + pid + chunk_start + offset, mask=mask)
        current_sum += chunk_data.sum()
        elements_loaded += BLOCK_SIZE

    tl.store(block_output_ptr + pid, current_sum)


# input, output are tensors on the GPU
def solve(
    input: torch.Tensor, output: torch.Tensor, 
    N: int, window_size: int
):
    BLOCK_SIZE = 1024
    grid = ( N - window_size + 1, )
    
    block_sub_sum = torch.empty( [N - window_size + 1], device=input.device, dtype=input.dtype)
    max_subarray_kernel[grid](
        input, block_sub_sum, 
        N, window_size,
        BLOCK_SIZE
    )
    
    output.copy_(block_sub_sum.max())
    
