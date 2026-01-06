import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    alpha: tl.constexpr, beta: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_res_row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offsets_res_col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    offsets_res = offsets_res_row * N + offsets_res_col
    mask_res = (offsets_res_row < M) & (offsets_res_col < N)

    offsets_a_row_start = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offsets_a_col = tl.arange(0, BLOCK_K)[None, :]
    
    offsets_b_row = tl.arange(0, BLOCK_K)[:, None]
    offsets_b_col_start = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        offsets_a_block = offsets_a_row_start * K + (k + offsets_a_col)
        offsets_b_block = (k + offsets_b_row) * N + offsets_b_col_start
        mask_a_block = (offsets_a_row_start < M) & ((k + offsets_a_col) < K)
        mask_b_block = ((k + offsets_b_row) < K) & (offsets_b_col_start < N)

        a_block = tl.load(A_ptr + offsets_a_block, mask=mask_a_block, other=0.0)
        b_block = tl.load(B_ptr + offsets_b_block, mask=mask_b_block, other=0.0)

        acc += tl.dot(a_block, b_block)

    c_block = tl.load(C_ptr + offsets_res, mask=mask_res, other=0.0)
    c_block = alpha * acc + beta * c_block
    tl.store(C_ptr + offsets_res, c_block, mask=mask_res)

# a, b, c are tensors on the GPU
def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    gemm_kernel[(grid_m, grid_n)](
        A, B, C,
        M, N, K,
        alpha, beta,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
