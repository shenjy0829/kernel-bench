import torch
import triton
import triton.language as tl

@triton.jit
def softmax_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr, 
    M, N, d,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_d: tl.constexpr
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_d = tl.arange(0, BLOCK_SIZE_d)

    ## load Q block
    offsets_q_block_m = pid * BLOCK_SIZE_M + offs_m[:, None]
    mask_q_block_m = offsets_q_block_m < M
    offsets_q_block_d = offs_d[None, :]
    mask_q_block_d = offs_d < d
    
    offsets_q_block = offsets_q_block_m * d + offsets_q_block_d
    mask_q_block = mask_q_block_m & mask_q_block_d

    q_block = tl.load(
        q_ptr + offsets_q_block,
        mask = mask_q_block,
        other = 0.0
    ) # [BLOCK_SIZE_M, BLOCK_SIZE_d]

    ## online softmax attention variables
    attention_logit_scale = 1.0 / tl.sqrt(d + 0.0)
    online_softmax_running_sum = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)                  # store running sum of softmax(score)
    online_softmax_current_max = tl.full([BLOCK_SIZE_M], float("-inf"), dtype=tl.float32)    # store running max of softmax(score)
    online_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_d), dtype=tl.float32)             # store running softmax(score) @ V

    # loop over N dimension of K, V
    for start_n in range(0, N, BLOCK_SIZE_N):
        offsets_kv_block_n = start_n + offs_n[:, None]
        mask_kv_block_n = offsets_kv_block_n < N
        offsets_kv_block_d = offs_d[None, :]
        mask_kv_block_d = offs_d < d

        offsets_kv_block = offsets_kv_block_n * d + offsets_kv_block_d
        mask_kv_block = mask_kv_block_n & mask_kv_block_d

        k_block = tl.load(
            k_ptr + offsets_kv_block,
            mask = mask_kv_block,
            other = 0.0
        ) # [BLOCK_SIZE_N, BLOCK_SIZE_d]

        v_block = tl.load(
            v_ptr + offsets_kv_block,
            mask = mask_kv_block,
            other = 0.0
        ) # [BLOCK_SIZE_N, BLOCK_SIZE_d]

        # compute attention scores
        attn_scores = tl.dot(q_block, tl.trans(k_block)) * attention_logit_scale
        attn_scores_mask = mask_q_block_m & (start_n + offs_n[None, :] < N)
        attn_scores = tl.where(attn_scores_mask, attn_scores, float('-inf'))

        # online softmax attn
        current_max = tl.max(attn_scores, axis=1)                       # [BLOCK_SIZE_M]
        new_max = tl.maximum(online_softmax_current_max, current_max)   # [BLOCK_SIZE_M]
        exp_diff_prev = tl.exp(online_softmax_current_max - new_max)    # [BLOCK_SIZE_M]
        online_softmax_current_max = new_max  # [BLOCK_SIZE_M]

        attn_scores_shifted = attn_scores - new_max[:, None]   # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        attn_scores_shifted_exp = tl.exp(attn_scores_shifted)  # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        online_softmax_running_sum = tl.fma(online_softmax_running_sum, exp_diff_prev, tl.sum(attn_scores_shifted_exp, axis=1))  # [BLOCK_SIZE_M]
        # online_softmax_running_sum [BLOCK_SIZE_M]

        online_accumulator = tl.fma(online_accumulator, exp_diff_prev[:, None], tl.dot(attn_scores_shifted_exp, v_block))          # [BLOCK_SIZE_M, BLOCK_SIZE_d]
        # online_accumulator [BLOCK_SIZE_M, BLOCK_SIZE_d]

    online_accumulator = online_accumulator / online_softmax_running_sum[:, None]

    tl.store(
        output_ptr + offsets_q_block,
        online_accumulator,
        mask = mask_q_block
    )

# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,        # M x d
    K: torch.Tensor,        # N x d
    V: torch.Tensor,        # N x d
    output: torch.Tensor, 
    M: int, 
    N: int, 
    d: int
):
    # M 决定了输出多少个结果，
    # N 决定了每个结果参考了多少输入信息。
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_d = max(128, triton.next_power_of_2(d))
    grid = (triton.cdiv(M, BLOCK_SIZE_M), 1)

    softmax_attention_kernel[grid](Q, K, V, output,
                                    M, N, d,
                                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_d)
    
