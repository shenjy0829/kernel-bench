import torch
import triton
import triton.language as tl

@triton.jit 
def mha_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    N, d_model, h,
    BLOCKSIZE_N: tl.constexpr, 
    BLOCKSIZE_d: tl.constexpr
):
    pid0 = tl.program_id(0)             # query block index
    pid1 = tl.program_id(1)             # head index
    offs_n = tl.arange(0, BLOCKSIZE_N)
    offs_d = tl.arange(0, BLOCKSIZE_d)

    d_head = d_model // h

    # load Q block
    offsets_n = pid0 * BLOCKSIZE_N + offs_n[:, None]
    offsets_dim_head = pid1 * d_head + offs_d[None, :]
    mask_q_n = offsets_n < N
    mask_dim_head = offs_d < d_head

    offsets_block = offsets_n * d_model + offsets_dim_head
    mask_block = mask_q_n & mask_dim_head

    block_q = tl.load(
        Q_ptr + offsets_block,
        mask = mask_block,
        other = 0.0
    ) # [BLOCKSIZE_N, BLOCKSIZE_d]

    # online para
    attention_logit_scale = 1.0 / tl.sqrt(d_head + 0.0)
    online_softmax_running_sum = tl.zeros([BLOCKSIZE_N], dtype=tl.float32)                  # store running sum of softmax(score)
    online_softmax_current_max = tl.full([BLOCKSIZE_N], float("-inf"), dtype=tl.float32)    # store running max of softmax(score)
    online_accumulator = tl.zeros((BLOCKSIZE_N, BLOCKSIZE_d), dtype=tl.float32)             # store running softmax(score) @ V

    # loop over N dimensions of K, V
    for start_n in range(0, N, BLOCKSIZE_N):
        offsets_k_n = start_n + offs_n[:, None]
        mask_k_n = offsets_k_n < N

        offsets_k = offsets_k_n * d_model + offsets_dim_head
        mask_k = mask_k_n & mask_dim_head

        block_k = tl.load(
            K_ptr + offsets_k,
            mask = mask_k,
            other = 0.0
        ) # [BLOCKSIZE_N, BLOCKSIZE_d]

        offsets_v = offsets_k_n * d_model + offsets_dim_head
        mask_v = mask_k_n & mask_dim_head

        block_v = tl.load(
            V_ptr + offsets_v,
            mask = mask_v,
            other = 0.0
        ) # [BLOCKSIZE_N, BLOCKSIZE_d]

        # Compute attention scores
        block_attn_scores = tl.dot(block_q, tl.trans(block_k)) * attention_logit_scale          # [BLOCKSIZE_N, BLOCKSIZE_N]
        block_attn_scores_mask = mask_q_n & (start_n + tl.arange(0, BLOCKSIZE_N)[None, :] < N)  # 注意这里结果的 mask 和 mask_k_n 不一样
        block_attn_scores = tl.where(block_attn_scores_mask, block_attn_scores, float("-inf"))

        # Online softmax
        current_max = tl.max(block_attn_scores, axis=1)                 # [BLOCKSIZE_N]
        new_max = tl.maximum(online_softmax_current_max, current_max)
        exp_diff_prev = tl.exp(online_softmax_current_max - new_max) 
        online_softmax_current_max = new_max  # [BLOCKSIZE_N] 

        block_attn_scores_shifted = block_attn_scores - new_max[:, None]  # [BLOCKSIZE_N, BLOCKSIZE_N]
        block_attn_scores_shifted_exp = tl.exp(block_attn_scores_shifted)  # [BLOCKSIZE_N, BLOCKSIZE_N]
        online_softmax_running_sum = tl.fma(online_softmax_running_sum, exp_diff_prev, tl.sum(block_attn_scores_shifted_exp, axis=1))  # [BLOCKSIZE_N]
        
        online_accumulator = tl.fma(online_accumulator, exp_diff_prev[:, None], tl.dot(block_attn_scores_shifted_exp, block_v))          # [BLOCKSIZE_N, BLOCKSIZE_d]

    # div online_accumulator by online_softmax_running_sum
    online_accumulator = online_accumulator / online_softmax_running_sum[:, None]

    # store ouput     
    tl.store(
        output_ptr + offsets_block,
        online_accumulator,
        mask = mask_block
    )

# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor,        # [N, d_model]
    K: torch.Tensor,        # [N, d_model]
    V: torch.Tensor,        # [N, d_model]
    output: torch.Tensor,
    N: int,
    d_model: int,           # d_head * h 
    h: int,
):
    # Q K V [N, d_head, h] --> [h, N, d_head]
    # Q @ K^T / sqrt(d_model) [h, N, N]
    # Softmax [h, N, N]
    # Softmax @ V [h, N, d_head] --> [N, d_model]
    
    BLOCKSIZE_N = 32
    BLOCKSIZE_d = max(16, d_model // h) # BLOCKSIZE_d >= d_head

    grid = (triton.cdiv(N, BLOCKSIZE_N), h)
    # each block process [BLOCKSIZE_N, BLOCKSIZE_d] 事实上就是 [BLOCKSIZE_N, 1个head] 
    # BLOCKSIZE_d 是为了适配 triton 的 vector load/store
    mha_kernel[grid](Q, K, V, output,
                     N, d_model, h,
                     BLOCKSIZE_N, BLOCKSIZE_d,
                     num_warps=4)


