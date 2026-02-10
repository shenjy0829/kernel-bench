import torch
import triton
import triton.language as tl

@triton.jit
def conv3d_kernel(
    input_ptr: torch.Tensor, kernel_ptr: torch.Tensor, output_ptr: torch.Tensor,
    input_depth: int, input_rows: int, input_cols: int,
    kernel_depth: int, kernel_rows: int, kernel_cols: int,
    output_depth: int, output_rows: int, output_cols: int,
    BLOCKSIZE_d: tl.constexpr, BLOCKSIZE_r: tl.constexpr, BLOCKSIZE_c: tl.constexpr
):
    pid_d = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)
    offs_d = tl.arange(0, BLOCKSIZE_d)
    offs_r = tl.arange(0, BLOCKSIZE_r)
    offs_c = tl.arange(0, BLOCKSIZE_c)

    offsets_d = pid_d * BLOCKSIZE_d + offs_d[:, None, None]
    offsets_r = pid_r * BLOCKSIZE_r + offs_r[None, :, None]
    offsets_c = pid_c * BLOCKSIZE_c + offs_c[None, None, :]

    mask_d = offsets_d < output_depth
    mask_r = offsets_r < output_rows
    mask_c = offsets_c < output_cols

    offsets_block_result = offsets_d * output_rows * output_cols + offsets_r * output_cols + offsets_c
    mask_block_result = mask_d & mask_r & mask_c


    block_result = tl.zeros((BLOCKSIZE_d, BLOCKSIZE_r, BLOCKSIZE_c), dtype=tl.float32)
    
    for kd in range(kernel_depth):
        for kr in range(kernel_rows):
            for kc in range(kernel_cols):
                offsets_input_d = offsets_d + kd
                offsets_input_r = offsets_r + kr
                offsets_input_c = offsets_c + kc
                offsets_input = offsets_input_d * input_rows * input_cols + offsets_input_r * input_cols + offsets_input_c

                mask_input_d = offsets_input_d < input_depth
                mask_input_r = offsets_input_r < input_rows
                mask_input_c = offsets_input_c < input_cols
                mask_input = mask_input_d & mask_input_r & mask_input_c

                block_input = tl.load(
                    input_ptr + offsets_input,
                    mask=mask_input,
                    other=0.0
                )

                kernel_offset = kd * kernel_rows * kernel_cols + kr * kernel_cols + kc
                kernel_value = tl.load(kernel_ptr + kernel_offset)
                block_result += block_input * kernel_value

    tl.store(output_ptr + offsets_block_result, block_result, mask=mask_block_result)


# input, kernel, output are tensors on the GPU
def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_depth: int,
    input_rows: int,
    input_cols: int,
    kernel_depth: int,
    kernel_rows: int,
    kernel_cols: int,
):
    BLOCKSIZE_d = 4
    BLOCKSIZE_r = 4
    BLOCKSIZE_c = 128
    # results [input_depth - kernel_depth + 1, input_rows - kernel_rows + 1, input_cols - kernel_cols + 1]
    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    grid = (triton.cdiv(output_depth, BLOCKSIZE_d), 
            triton.cdiv(output_rows, BLOCKSIZE_r), 
            triton.cdiv(output_cols, BLOCKSIZE_c))
    conv3d_kernel[grid](
        input, kernel, output,
        input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols,
        output_depth, output_rows, output_cols,
        BLOCKSIZE_d, BLOCKSIZE_r, BLOCKSIZE_c,
    )