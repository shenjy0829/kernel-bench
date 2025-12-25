import torch
import triton
import triton.language as tl

@triton.jit
def rms_kernel(
    input_ptr, sum_x2_ptr, 
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(sum_x2_ptr, sum_x2)

@triton.jit
def norm_kernel(
    input_ptr, output_ptr, 
    sum_x2_ptr,
    N, gamma, beta, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sum_x2 = tl.load(sum_x2_ptr)
    rms = tl.sqrt(sum_x2 / N + eps)
    output = gamma * input / rms + beta

    tl.store(output_ptr + offsets, output, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: float, beta: float, 
          output: torch.Tensor, N: int, eps: float):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE), )

    sum_x2 = torch.zeros(1, device=input.device, dtype=input.dtype)
    rms_kernel[grid](
        input, sum_x2, 
        N, 
        BLOCK_SIZE
    )

    norm_kernel[grid](
        input, output, 
        sum_x2,
        N, gamma, beta, eps,
        BLOCK_SIZE
    )