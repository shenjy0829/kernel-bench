import torch
import triton
import triton.language as tl

@triton.jit
def mse_kernel(
    predictions_ptr, targets_ptr, mse_ptr, 
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    input_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offsets < N
    
    input_pred = tl.load( predictions_ptr + input_offsets, mask = input_mask, other=0.0)
    input_tgt = tl.load( targets_ptr + input_offsets, mask = input_mask, other=0.0)

    diff = input_pred - input_tgt
    sq_diff = diff * diff
    sum_sq_diff = tl.sum(sq_diff, axis=0)

    tl.atomic_add(mse_ptr, sum_sq_diff / N)

# predictions, targets, mse are tensors on the GPU
def solve(predictions: torch.Tensor, targets: torch.Tensor, mse: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    grid = (num_blocks,)

    mse.zero_()
    mse_kernel[grid](
        predictions, targets, mse,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    