import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input, kernel, output,
    input_rows, input_cols,
    kernel_rows, kernel_cols,
    output_rows, output_cols,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_offsets = (pid_row * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW))[:, None]
    col_offsets = (pid_col * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL))[None, :]
    offsets = row_offsets * output_cols + col_offsets

    row_mask = row_offsets < output_rows
    col_mask = col_offsets < output_cols
    mask = row_mask & col_mask

    block_result = tl.zeros((BLOCK_SIZE_ROW, BLOCK_SIZE_COL), dtype=tl.float32)

    # dot product between the kernel and the corresponding input patch
    for kr in range(kernel_rows):
        for kc in range(kernel_cols):
            input_row_offsets = row_offsets + kr
            input_col_offsets = col_offsets + kc
            input_block_offsets = input_row_offsets * input_cols + input_col_offsets

            input_row_mask = input_row_offsets < input_rows
            input_col_mask = input_col_offsets < input_cols
            input_block_mask = input_row_mask & input_col_mask

            input_vals = tl.load(
                input + input_block_offsets,
                mask=input_block_mask,
                other=0.0
            )
            kernel_val = tl.load(kernel + kr * kernel_cols + kc)
            block_result += input_vals * kernel_val

    tl.store(output + offsets, block_result, mask=mask)


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, 
    input_rows: int, input_cols: int, 
    kernel_rows: int, kernel_cols: int):
    
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    BLOCK_SIZE_ROW = 32
    BLOCK_SIZE_COL = 32
    grid = (triton.cdiv(output_rows, BLOCK_SIZE_ROW), triton.cdiv(output_cols, BLOCK_SIZE_COL))
    conv2d_kernel[grid](
        input, kernel, output,
        input_rows, input_cols,
        kernel_rows, kernel_cols,
        output_rows, output_cols,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW, BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )