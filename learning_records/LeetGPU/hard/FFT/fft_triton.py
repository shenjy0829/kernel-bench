import torch
import triton
import triton.language as tl
import math

@triton.jit
def butterfly_kernel(
    X_real_ptr, X_imag_ptr, 
    Y_real_ptr, Y_imag_ptr,
    stride, size,
    BLOCK_SIZE: tl.constexpr
):
    """
    单个蝶形运算阶段的 Kernel。
    实际上由 solve 函数调度多个 Kernel 来完成完整的 FFT。
    
    参数:
    stride: 当前蝶形运算的跨度 (1, 2, 4, ..., N/2)
    size: 2 * stride (当前子问题的规模)
    """
    pid = tl.program_id(0)
    # 每个线程处理一对数据 (k, k + stride)
    # 全局索引 i
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 确定我们在哪个蝶形组
    # 每个组大小为 size, 其中包含 size/2 个蝶形对
    # 逻辑索引 k: 0 ~ stride-1 (旋转因子索引)
    
    # 将线性索引 i 映射到蝶形索引
    # group_id = i // stride
    # offset_in_group = i % stride
    # 实际索引 pos = group_id * size + offset_in_group
    
    group_id = i // stride
    offset = i % stride
    
    idx_1 = group_id * (stride * 2) + offset
    idx_2 = idx_1 + stride
    
    # 加载两个点
    # 注意：这里假设输入已经是按位反转顺序或者我们在做 Decimation-in-Frequency
    # 这里演示 Cooley-Tukey 的基本蝶形
    
    real1 = tl.load(X_real_ptr + idx_1)
    imag1 = tl.load(X_imag_ptr + idx_1)
    
    real2 = tl.load(X_real_ptr + idx_2)
    imag2 = tl.load(X_imag_ptr + idx_2)
    
    # 计算旋转因子 (Twiddle Factors)
    # W_N^k = exp(-j * 2 * pi * k / (2 * stride))
    # k = offset
    # N_stage = 2 * stride
    
    angle = -2.0 * 3.141592653589793 * offset / (stride * 2.0)
    w_real = tl.cos(angle)
    w_imag = tl.sin(angle)
    
    # 蝶形运算
    # out1 = in1 + in2 * W
    # out2 = in1 - in2 * W
    # 但是具体形式取决于输入是时域抽取还是频域抽取
    
    # 这里实现 Cooley-Tukey Decimation-in-Time (DIT)
    # 需要输入是 bit-reversed，或者在这一步做 bit-reversal
    # 为了简化，假设这是标准的递归合并步骤：
    # 上一级输出已经是 DFT_L 和 DFT_R
    # Out[k] = L[k] + W*R[k]
    # Out[k+stride] = L[k] - W*R[k]
    
    # 复数乘法: in2 * W
    t_real = real2 * w_real - imag2 * w_imag
    t_imag = real2 * w_imag + imag2 * w_real
    
    y1_real = real1 + t_real
    y1_imag = imag1 + t_imag
    
    y2_real = real1 - t_real
    y2_imag = imag1 - t_imag
    
    tl.store(Y_real_ptr + idx_1, y1_real)
    tl.store(Y_imag_ptr + idx_1, y1_imag)
    tl.store(Y_real_ptr + idx_2, y2_real)
    tl.store(Y_imag_ptr + idx_2, y2_imag)


def bit_reverse_copy(X_real, X_imag, rev_real, rev_imag, N):
    # 简单的位反转置换，用于 DIT FFT 的预处理
    # 实际生产中这也应该是一个 Kernel
    idx = torch.arange(N, device=X_real.device)
    # 计算位反转索引
    rev_idx = torch.zeros_like(idx)
    num_bits = N.bit_length() - 1
    for i in range(num_bits):
        rev_idx = (rev_idx << 1) | (idx & 1)
        idx >>= 1
    
    rev_real[:] = X_real[rev_idx]
    rev_imag[:] = X_imag[rev_idx]

# signal and spectrum are tensors on the GPU
def solve(signal: torch.Tensor, spectrum: torch.Tensor, N: int):
    """
    signal: [N], complex (real part in signal.real, imag in signal.imag if complex tensor)
            或者假设 signal 是实数输入，spectrum 是复数输出
    注意：Triton 原生不支持 complex 类型，通常拆分为 real/imag 指针
    """
    
    # 确保 N 是 2 的幂
    assert (N & (N-1) == 0) and N > 0, "N must be a power of 2"
    
    # 预处理：将 Tensor 转换为实部和虚部
    if signal.is_complex():
        X_real = signal.real.contiguous()
        X_imag = signal.imag.contiguous()
    else:
        X_real = signal.contiguous()
        X_imag = torch.zeros_like(signal)

    # 结果 Buffer (Ping-Pong buffers)
    buf_real = torch.empty_like(X_real)
    buf_imag = torch.empty_like(X_imag)
    
    # Step 1: Bit-Reversal Permutation
    # 将输入数据按位反转顺序拷贝到 buf 中作为起点
    bit_reverse_copy(X_real, X_imag, buf_real, buf_imag, N)
    
    # 迭代 logN 级
    num_stages = int(math.log2(N))
    
    # Ping-Pong Buffer 指针
    curr_real, curr_imag = buf_real, buf_imag
    next_real, next_imag = torch.empty_like(X_real), torch.empty_like(X_imag)
    
    BLOCK_SIZE = 128 # 配合 strides
    
    for stage in range(num_stages):
        stride = 1 << stage # 1, 2, 4, ...
        # size = stride * 2
        
        # 启动 Kernel 处理这一级
        # 总共有 N/2 个蝶形运算
        grid = (triton.cdiv(N // 2, BLOCK_SIZE), )
        
        butterfly_kernel[grid](
            curr_real, curr_imag,
            next_real, next_imag,
            stride, stride * 2,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Swap buffers
        curr_real, next_real = next_real, curr_real
        curr_imag, next_imag = next_imag, curr_imag

    # 将最终结果写入 spectrum
    # spectrum 应该是 complex64/128
    if spectrum.is_complex():
        spectrum.real[:] = curr_real# filepath: /home/roxy-0x00/Desktop/escape_101/Sliverfish_Repo/kernel-bench/learning_records/LeetGPU/hard/FFT/fft_triton.py
