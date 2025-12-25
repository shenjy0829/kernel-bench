import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    A_ptr, B_ptr, C_ptr, 
    M, N, K,
    stride_am, stride_an, 
    stride_bn, stride_bk, 
    stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # each kernel process
    # a -- BLOCK_SIZE_M * N
    # b -- N *BLOCK_SIZE_K
    # c -- BLOCK_SIZE_M * BLOCK_SIZE_K
      
    A_ptr = A_ptr.to(tl.pointer_type(tl.float32))
    B_ptr = B_ptr.to(tl.pointer_type(tl.float32))
    C_ptr = C_ptr.to(tl.pointer_type(tl.float32))

    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask_m = offsets_m < M
    mask_k = offsets_k < K

    offsets_block_c = C_ptr + offsets_m[:,None] * stride_cm + offsets_k[None, :] * stride_ck
    mask_block_c = mask_m[:, None] & mask_k[None, :]

    block_c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in tl.range(0, N, BLOCK_SIZE_N):
        offsets_n = n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offsets_n < N
        offsets_block_a = A_ptr + offsets_m[:,None] * stride_am + offsets_n[None, :] * stride_an
        offsets_block_b = B_ptr + offsets_n[:,None] * stride_bn + offsets_k[None, :] * stride_bk
        mask_block_a = mask_m[:, None] & mask_n[None, :]
        mask_block_b = mask_n[:, None] & mask_k[None, :]
        
        block_a = tl.load(offsets_block_a, mask=mask_block_a, other = 0)
        block_b = tl.load(offsets_block_b, mask=mask_block_b, other = 0)
        block_c += tl.dot(block_a, block_b)
    tl.store( offsets_block_c, block_c, mask=mask_block_c)


# a, b, c are tensors on the GPU
# a -- M * N
# b -- N * K
# c -- M * K 
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1 
    stride_bn, stride_bk = K, 1  
    stride_cm, stride_ck = K, 1

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_K)) 
    matrix_multiplication_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )