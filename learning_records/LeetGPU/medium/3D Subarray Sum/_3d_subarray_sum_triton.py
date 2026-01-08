import torch
import triton
import triton.language as tl

@triton.jit
def subarray_sum_3d_kernel(
    input_ptr, output_ptr, 
    stride_n, stride_m, stride_k,
    S_DEP, E_DEP,
    S_ROW, E_ROW,
    S_COL, E_COL,
    BLOCK_SIZE_DEP: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr
):
    pid_dep = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_col = tl.program_id(2)

    dep_offsets = S_DEP + pid_dep * BLOCK_SIZE_DEP + tl.arange(0, BLOCK_SIZE_DEP)[:, None, None]
    row_offsets = S_ROW + pid_row * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)[None, :, None]
    col_offsets = S_COL + pid_col * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)[None, None, :]

    dep_mask = dep_offsets < (E_DEP + 1)
    row_mask = row_offsets < (E_ROW + 1)
    col_mask = col_offsets < (E_COL + 1)
    mask = dep_mask & row_mask & col_mask

    input = tl.load(
        input_ptr + dep_offsets * stride_n + row_offsets * stride_m + col_offsets * stride_k,
        mask=mask,
        other=0.0
    )
    block_sum = tl.sum(input)
    tl.atomic_add(output_ptr, block_sum)

def solve(
    input: torch.Tensor, output: torch.Tensor, 
    N: int, M: int, K: int, 
    S_DEP: int, E_DEP: int, 
    S_ROW: int, E_ROW: int, 
    S_COL: int, E_COL: int
):
    stride_n, stride_m, stride_k = input.stride()
    
    BLOCK_SIZE_DEP = 32
    BLOCK_SIZE_ROW = 32
    BLOCK_SIZE_COL = 32
    
    grid_deps = triton.cdiv(E_DEP - S_DEP + 1, BLOCK_SIZE_DEP)
    grid_rows = triton.cdiv(E_ROW - S_ROW + 1, BLOCK_SIZE_ROW)
    grid_cols = triton.cdiv(E_COL - S_COL + 1, BLOCK_SIZE_COL)
    grid = (grid_deps, grid_rows, grid_cols)

    output.zero_()
    subarray_sum_3d_kernel[grid](
        input, output,
        stride_n, stride_m, stride_k,
        S_DEP, E_DEP,
        S_ROW, E_ROW,
        S_COL, E_COL,
        BLOCK_SIZE_DEP=BLOCK_SIZE_DEP,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )