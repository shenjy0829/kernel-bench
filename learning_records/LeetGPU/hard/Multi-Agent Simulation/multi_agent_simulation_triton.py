import torch
import triton
import triton.language as tl

@triton.jit
def multi_agent_kernel_decoupled(
    agents_ptr, agents_next_ptr, 
    N,
    BLOCK_M: tl.constexpr,  # 处理多少个 Agent i (输出行数)
    BLOCK_N: tl.constexpr   # 每次加载多少个 Neighbor j (输入列数)
):
    pid = tl.program_id(0)
    # i 维度：使用较小的分块以减少 Reduction 误差
    offs_i = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_i = offs_i < N

    ptr_i = agents_ptr + offs_i * 4
    
    # 保持 float32 以匹配 PyTorch 行为
    x_i = tl.load(ptr_i + 0, mask=mask_i, other=0.0)
    y_i = tl.load(ptr_i + 1, mask=mask_i, other=0.0)
    vx_i = tl.load(ptr_i + 2, mask=mask_i, other=0.0)
    vy_i = tl.load(ptr_i + 3, mask=mask_i, other=0.0)

    # 累加器
    acc_neighbor_count = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_sum_vx = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_sum_vy = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    r = 5.0
    r2 = r * r 

    # j 循环：使用较大的步长 (BLOCK_N) 以优化显存带宽
    # 这里 BLOCK_N 可以是 128, 256 甚至 512
    for start_j in range(0, N, BLOCK_N):
        offs_j = start_j + tl.arange(0, BLOCK_N)
        mask_j = offs_j < N
        ptr_j = agents_ptr + offs_j * 4
        
        # 加载 j 数据：一次加载一大块
        x_j = tl.load(ptr_j + 0, mask=mask_j, other=0.0)
        y_j = tl.load(ptr_j + 1, mask=mask_j, other=0.0)
        vx_j = tl.load(ptr_j + 2, mask=mask_j, other=0.0)
        vy_j = tl.load(ptr_j + 3, mask=mask_j, other=0.0)
        
        # 计算距离 (Broadcast: [BLOCK_M, 1] - [1, BLOCK_N])
        # Triton 会自动处理这种广播，生成 [BLOCK_M, BLOCK_N] 的中间结果
        dx = x_j[None, :] - x_i[:, None]
        dy = y_j[None, :] - y_i[:, None]
        dist_sq = dx * dx + dy * dy
        
        in_range = dist_sq < r2
        idx_i = offs_i[:, None]
        idx_j = offs_j[None, :]
        not_self = idx_i != idx_j
        
        valid_mask = in_range & not_self & mask_j[None, :]
        valid_f = valid_mask.to(tl.float32)
        
        # 累加：沿着 axis=1 (N维度) 规约
        # 这里的 sum 是把 BLOCK_N 个邻居的影响加起来
        acc_neighbor_count += tl.sum(valid_f, axis=1)
        acc_sum_vx += tl.sum(valid_f * vx_j[None, :], axis=1)
        acc_sum_vy += tl.sum(valid_f * vy_j[None, :], axis=1)

    # 显式 FP32
    alpha = tl.full([1], 0.05, dtype=tl.float32)

    has_neighbors = acc_neighbor_count > 0
    safe_count = tl.where(has_neighbors, acc_neighbor_count, 1.0)
    
    avg_vx = acc_sum_vx / safe_count
    avg_vy = acc_sum_vy / safe_count
    
    # Update logic matching PyTorch exact operations order
    diff_vx = avg_vx - vx_i
    diff_vy = avg_vy - vy_i
    
    new_vx = vx_i + alpha * diff_vx
    new_vy = vy_i + alpha * diff_vy
    
    final_vx = tl.where(has_neighbors, new_vx, vx_i)
    final_vy = tl.where(has_neighbors, new_vy, vy_i)
    
    new_x = x_i + final_vx
    new_y = y_i + final_vy
    
    ptr_next = agents_next_ptr + offs_i * 4
    tl.store(ptr_next + 0, new_x, mask=mask_i)
    tl.store(ptr_next + 1, new_y, mask=mask_i)
    tl.store(ptr_next + 2, final_vx, mask=mask_i)
    tl.store(ptr_next + 3, final_vy, mask=mask_i)

def solve(agents: torch.Tensor, agents_next: torch.Tensor, N: int):
    # BLOCK_M (Output Tile): 保持小 (32)，为了精度和 Reference 一致
    # 32 是一个 Warp 的大小，规约行为最稳定
    BLOCK_M = 64
    
    # BLOCK_N (Input Tile): 保持大 (128 或 256)，为了吃满显存带宽
    BLOCK_N = 1024
    
    # Grid 只需要由输出数量 N 和 BLOCK_M 决定
    grid = (triton.cdiv(N, BLOCK_M), )
    
    multi_agent_kernel_decoupled[grid](
        agents, agents_next,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )