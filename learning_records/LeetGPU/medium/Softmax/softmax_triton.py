import torch
import triton
import triton.language as tl


@triton.jit
def softmax_reduce_kernel(
    input_ptr, 
    output_max_ptr, output_sum_ptr,  
    N: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    
    x = tl.load(input_ptr + offset, mask=mask, other=-float('inf'))
    
    m = tl.max(x, axis=0)
    
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    
    tl.store(output_max_ptr + pid, m)
    tl.store(output_sum_ptr + pid, s)

@triton.jit
def combine_results_kernel(
    input_max_ptr, input_sum_ptr, 
    global_max_ptr, global_sum_ptr,
    num_blocks: tl.constexpr
):
    global_max = float('-inf')
    global_sum = 0.0
    
    for i in tl.range(num_blocks):
        max_i = tl.load(input_max_ptr + i)
        sum_i = tl.load(input_sum_ptr + i)
        
        new_global_max = tl.maximum(global_max, max_i)
        global_sum = global_sum * tl.exp(global_max - new_global_max) + \
                   sum_i * tl.exp(max_i - new_global_max)
        
        global_max = new_global_max
        
    tl.store(global_max_ptr, global_max)
    tl.store(global_sum_ptr, global_sum)


@triton.jit
def normalize_kernel(
    input_ptr, output_ptr, 
    global_max_ptr, global_sum_ptr, 
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    
    input = tl.load(input_ptr + offset, mask=mask, other=-float('inf'))
    
    global_max = tl.load(global_max_ptr)
    global_sum = tl.load(global_sum_ptr)
    
    output = tl.exp(input - global_max) / global_sum
    tl.store(output_ptr + offset, output, mask=mask)

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 4096 
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    partial_max = torch.empty(num_blocks, device=input.device, dtype=torch.float32)
    partial_sum = torch.empty(num_blocks, device=input.device, dtype=torch.float32)
    global_max = torch.empty(1, device=input.device, dtype=torch.float32)
    global_sum = torch.empty(1, device=input.device, dtype=torch.float32)
    
    # get block max and sum
    softmax_reduce_kernel[(num_blocks,)](
        input, 
        partial_max, partial_sum, 
        N, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # combine block results
    combine_results_kernel[(1,)](
        partial_max, partial_sum, 
        global_max, global_sum,
        num_blocks=num_blocks
    )
    
    # normalize
    normalize_kernel[(num_blocks,)](
        input, output, 
        global_max, global_sum,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )

# 1. 分 block 计算 partial max 和 partial sum
# 2. 一个 thread 计算 global max 和 global sum
# 3. 分 block 计算归一化结果