import torch
import triton
import triton.language as tl

@triton.jit
def multi_agent_kernel(
    agents_ptr, agents_next_ptr, 
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs_i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_i = offs_i < 4 * N

    ptr_i = agents_ptr + offs_i * 4
    
    # load agent i state
    x_i = tl.load(ptr_i + 0, mask=mask_i, other=0.0)
    y_i = tl.load(ptr_i + 1, mask=mask_i, other=0.0)
    vx_i = tl.load(ptr_i + 2, mask=mask_i, other=0.0)
    vy_i = tl.load(ptr_i + 3, mask=mask_i, other=0.0)

    acc_neighbor_count = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    acc_sum_vx = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    acc_sum_vy = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
    
    r = 5.0
    r2 = r * r

    for start_j in range(0, N, BLOCK_SIZE):
        offs_j = start_j + tl.arange(0, BLOCK_SIZE)
        mask_j = offs_j < N
        
        ptr_j = agents_ptr + offs_j * 4
        
        # load agent j state
        x_j = tl.load(ptr_j + 0, mask=mask_j, other=0.0)
        y_j = tl.load(ptr_j + 1, mask=mask_j, other=0.0)
        vx_j = tl.load(ptr_j + 2, mask=mask_j, other=0.0)
        vy_j = tl.load(ptr_j + 3, mask=mask_j, other=0.0)
        
        # cal distance squared
        dx = x_j[None, :] - x_i[:, None]
        dy = y_j[None, :] - y_i[:, None]
        
        dist_sq = dx * dx + dy * dy
        
        # check in range
        in_range = dist_sq < r2
        
        # not self interaction
        idx_i = offs_i[:, None]
        idx_j = offs_j[None, :]
        not_self = idx_i != idx_j
        
        # combined mask: in range AND not self AND j not out of bounds
        valid_mask = in_range & not_self & mask_j[None, :]
        
        # convert to float for accumulation
        valid_f = valid_mask.to(tl.float64)
        vx_j_f64 = vx_j.to(tl.float64)
        vy_j_f64 = vy_j.to(tl.float64)
        
        # sum over j dimension (axis=1) to get statistics for each i
        acc_neighbor_count += tl.sum(valid_f, axis=1)
        acc_sum_vx += tl.sum(valid_f * vx_j_f64[None, :], axis=1)
        acc_sum_vy += tl.sum(valid_f * vy_j_f64[None, :], axis=1)

    alpha = 0.05

    has_neighbors = acc_neighbor_count > 0
    
    vx_i_f64 = vx_i.to(tl.float64)
    vy_i_f64 = vy_i.to(tl.float64)

    avg_vx = tl.where(has_neighbors, acc_sum_vx / acc_neighbor_count, vx_i_f64)
    avg_vy = tl.where(has_neighbors, acc_sum_vy / acc_neighbor_count, vy_i_f64)
    
    new_vx = vx_i_f64 + alpha * (avg_vx - vx_i_f64)
    new_vy = vy_i_f64 + alpha * (avg_vy - vy_i_f64)
    
    new_x = x_i + new_vx.to(tl.float32)
    new_y = y_i + new_vy.to(tl.float32)
    
    ptr_next = agents_next_ptr + offs_i * 4
    tl.store(ptr_next + 0, new_x, mask=mask_i)
    tl.store(ptr_next + 1, new_y, mask=mask_i)
    tl.store(ptr_next + 2, new_vx.to(tl.float32), mask=mask_i)
    tl.store(ptr_next + 3, new_vy.to(tl.float32), mask=mask_i)


# agents, agents_next are tensors on the GPU
def solve(agents: torch.Tensor, agents_next: torch.Tensor, N: int):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE), )

    multi_agent_kernel[grid](
        agents, agents_next,
        N,
        BLOCK_SIZE
    )
    