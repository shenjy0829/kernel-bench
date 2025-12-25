import torch
import triton
import triton.language as tl

@triton.jit
def topk_kernel(input_ptr, top_k_block_ptr, 
                N, k,
                BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    input_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offsets < N

    input_data = tl.load(input_ptr + input_offsets, mask=input_mask, other=float("-inf"))
    output_data = tl.sort(input_data, descending=True)

    output_offsets = pid * k + tl.arange(0, BLOCK_SIZE)
    output_mask = tl.arange(0, BLOCK_SIZE) < k

    tl.store(top_k_block_ptr + output_offsets, output_data, mask=output_mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int, k: int):
    BLOCK_SIZE = 512

    grid = (triton.cdiv(N, BLOCK_SIZE), )

    top_k_block = torch.full([grid[0] * k], float("-inf"), device=input.device, dtype=torch.float32)
    topk_kernel[grid](input, top_k_block, N, k, BLOCK_SIZE)

    final_vals, _ = torch.topk(top_k_block, k) 
    output.copy_(final_vals)


