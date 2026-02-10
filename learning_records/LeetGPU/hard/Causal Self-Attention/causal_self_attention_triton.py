import torch
import triton
import triton.language as tl

@triton.jit 
def causal_attention_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    M, 
    d,
    BLOCKSIZE_M: tl.constexpr,
    BLOCKSIZE_d: tl.constexpr
):
    pid0 = tl.program_id(0) # query block index
    offs_m = tl.arange(0, BLOCKSIZE_M)
    offs_d = tl.arange(0, BLOCKSIZE_d)

    # load Q block
    offsets_m = pid0 * BLOCKSIZE_M + offs_m[:, None]
    offsets_dim_head = offs_d[None, :]
    mask_q_m = offsets_m < M
    mask_dim_head = offs_d < d
    

    offsets_block = offsets_m * d + offsets_dim_head
    mask_block = mask_q_m & mask_dim_head

    block_q = tl.load(
        Q_ptr + offsets_block,
        mask = mask_block,
        other = 0.0
    ) 

    # online para
    attention_logit_scale = 1.0 / tl.sqrt(d + 0.0)
    online_softmax_running_sum = tl.zeros([BLOCKSIZE_M], dtype=tl.float32)                  # store running sum of softmax(score)
    online_softmax_current_max = tl.full([BLOCKSIZE_M], float("-inf"), dtype=tl.float32)    # store running max of softmax(score)
    online_accumulator = tl.zeros((BLOCKSIZE_M, BLOCKSIZE_d), dtype=tl.float32)             # store running softmax(score) @ V

    # loop over M dimensions of K, V
    # Query 块索引是 pid0，对应的行号范围是 [pid0*BM, (pid0+1)*BM]
    # Key 块只需要遍历到 (pid0 + 1) * BLOCKSIZE_M 即可（包含对角线块）
    loop_limit = (pid0 + 1) * BLOCKSIZE_M
    loop_limit = tl.minimum(loop_limit, M) 
    for start_m in tl.range(0, loop_limit, BLOCKSIZE_M):
        range_indices = start_m + offs_m
        offsets_k_m = range_indices[:, None]
        mask_k_m = offsets_k_m < M

        offsets_k = offsets_k_m * d + offsets_dim_head
        mask_k = mask_k_m & mask_dim_head

        block_k = tl.load(
            K_ptr + offsets_k,
            mask = mask_k,
            other = 0.0
        )

        offsets_v = offsets_k_m * d + offsets_dim_head
        mask_v = mask_k_m & mask_dim_head

        block_v = tl.load(
            V_ptr + offsets_v,
            mask = mask_v,
            other = 0.0
        ) 

        # Compute attention scores
        block_attn_scores = tl.dot(block_q, tl.trans(block_k)) * attention_logit_scale          
        block_attn_scores_mask = mask_q_m & (range_indices[None, :] < M) 
        causal_mask = range_indices[None, :] <= offsets_m
        block_attn_scores = tl.where(block_attn_scores_mask & causal_mask, block_attn_scores, float("-inf"))
        
        # Online softmax
        current_max = tl.max(block_attn_scores, axis=1)                 
        new_max = tl.maximum(online_softmax_current_max, current_max)
        exp_diff_prev = tl.exp(online_softmax_current_max - new_max)  
        online_softmax_current_max = new_max

        block_attn_scores_shifted = block_attn_scores - new_max[:, None]  
        block_attn_scores_shifted_exp = tl.exp(block_attn_scores_shifted)  
        online_softmax_running_sum = tl.fma(online_softmax_running_sum, exp_diff_prev, tl.sum(block_attn_scores_shifted_exp, axis=1))  
          

        online_accumulator = tl.fma(online_accumulator, exp_diff_prev[:, None], tl.dot(block_attn_scores_shifted_exp, block_v))          

    # div online_accumulator by online_softmax_running_sum
    online_accumulator = online_accumulator / online_softmax_running_sum[:, None]

    # store ouput     
    tl.store(
        output_ptr + offsets_block,
        online_accumulator,
        mask = mask_block
    )

# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    
    BLOCKSIZE_M = 32
    BLOCKSIZE_d = max(16, triton.next_power_of_2(d))  

    grid = (triton.cdiv(M, BLOCKSIZE_M), )
    causal_attention_kernel[grid](Q, K, V, output,
                     M, d,
                     BLOCKSIZE_M, BLOCKSIZE_d)