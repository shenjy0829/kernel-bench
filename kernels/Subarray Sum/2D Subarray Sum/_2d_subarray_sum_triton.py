import torch
import triton
import triton.language as tl

@triton.jit
def subarray_sum_2d_kernel(
    input_ptr, output_ptr, 
    stride_n, stride_m,
    N, M,
    S_ROW, E_ROW,
    S_COL, E_COL,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_offsets = S_ROW + pid_row * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)[:, None]
    col_offsets = S_COL + pid_col * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)[None, :]

    row_mask = row_offsets < (E_ROW + 1)
    col_mask = col_offsets < (E_COL + 1)
    mask = row_mask & col_mask

    input = tl.load(
        input_ptr + row_offsets * stride_n + col_offsets * stride_m,
        mask=mask,
        other=0.0
    )
    block_sum = tl.sum(input)

    tl.atomic_add(output_ptr, block_sum)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, S_ROW: int, E_ROW: int, S_COL: int, E_COL: int):
    stride_n, stride_m = input.stride()
    
    BLOCK_SIZE_ROW = 16
    BLOCK_SIZE_COL = 16
    grid = (triton.cdiv(E_ROW - S_ROW + 1, BLOCK_SIZE_ROW), triton.cdiv(E_COL - S_COL + 1, BLOCK_SIZE_COL))

    output.zero_()
    subarray_sum_2d_kernel[grid](
        input, output,
        stride_n, stride_m,
        N, M,
        S_ROW, E_ROW,
        S_COL, E_COL,
        BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL
    )
