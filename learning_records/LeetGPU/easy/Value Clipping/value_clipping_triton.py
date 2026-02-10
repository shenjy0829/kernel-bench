import torch
import triton
import triton.language as tl


@triton.jit
def clip_kernel(input, output, lo, hi, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input + offset, mask=mask, other=0.0)
    x_clipped = tl.where(x < lo, lo, tl.where(x > hi, hi, x))
    tl.store(output + offset, x_clipped, mask=mask)


# input, output are tensors on the GPU
# clipping range is [lo, hi)
def solve(input: torch.Tensor, output: torch.Tensor, lo: float, hi: float, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    clip_kernel[grid](input, output, lo, hi, N, BLOCK_SIZE=BLOCK_SIZE)
