#!/usr/bin/env python3
"""
kernel_benchmark.py

Compare kernels for:
 - PyTorch (torch ops)
 - Triton (triton.jit kernels)
 - Native CUDA via CuPy RawKernel (fallback to numba.cuda if no cupy)

Benchmarks:
 - Elementwise add (vector/tensor add)
 - Matrix multiply (GEMM) [PyTorch vs Triton vs CuPy/Numba]

Usage:
  python kernel_benchmark.py
"""

import time
import sys
import math
import argparse
from functools import partial

import torch

# Try to import triton, cupy, numba in order; mark availability
HAS_TRITON = False
HAS_CUPY = False
HAS_NUMBA = False
triton = None
cupy = None
numba = None

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

try:
    import cupy as cupy
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

try:
    import numba
    from numba import cuda as nbcuda
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    print("No CUDA device detected. This script requires a CUDA GPU.")
    sys.exit(1)

# ---------- Helpers: timing wrappers ----------
def torch_time_fn(fn, warmup=5, iters=50):
    torch.cuda.synchronize()
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    return ms / iters  # avg ms

def cupy_time_fn(fn, warmup=5, iters=50):
    # uses cupy.cuda.Event
    for _ in range(warmup):
        fn()
    ev_start = cupy.cuda.Event()
    ev_end = cupy.cuda.Event()
    ev_start.record()
    for _ in range(iters):
        fn()
    ev_end.record()
    ev_end.synchronize()
    ms = cupy.cuda.get_elapsed_time(ev_start, ev_end)
    return ms / iters

def numba_time_fn(fn, warmup=5, iters=50):
    # use cuda.synchronize + time.perf_counter
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    total_ms = (end - start) * 1000.0
    return total_ms / iters

# ---------- Kernels ----------

# 1) PyTorch implementations
def pytorch_elem_add(a, b, out):
    out.copy_(a + b)

def pytorch_gemm(A, B, C):
    # C = A @ B (square matrices)
    torch.matmul(A, B, out=C)

# 2) Triton kernels (if available)
if HAS_TRITON:
    # Triton elementwise add
    @triton.jit
    def triton_elem_add_kernel(X_ptr, Y_ptr, Z_ptr, N: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * 1024 + tl.arange(0, 1024)
        mask = offs < N
        x = tl.load(X_ptr + offs, mask=mask)
        y = tl.load(Y_ptr + offs, mask=mask)
        tl.store(Z_ptr + offs, x + y, mask=mask)

    def triton_elem_add(a, b, out):
        # a,b,out are torch tensors on cuda (contiguous)
        N = a.numel()
        grid = ( (N + 1023) // 1024, )
        triton_elem_add_kernel[grid](a.data_ptr(), b.data_ptr(), out.data_ptr(), N)

    # Triton GEMM: block-based matmul (simple)
    @triton.jit
    def triton_gemm_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
                           stride_am: tl.constexpr, stride_ak: tl.constexpr,
                           stride_bk: tl.constexpr, stride_bn: tl.constexpr,
                           stride_cm: tl.constexpr, stride_cn: tl.constexpr):
        # block sizes
        BM = 128
        BN = 128
        BK = 32
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        A_block_ptr = A_ptr + offs_m[:, None] * stride_am + tl.arange(0, K)[None, :] * stride_ak
        B_block_ptr = B_ptr + tl.arange(0, K)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            k_range = tl.arange(0, BK)
            a = tl.load(A_ptr + (offs_m[:, None] * stride_am) + (k + k_range)[None, :] * stride_ak, mask=(offs_m[:, None] < M) & ((k + k_range)[None, :] < K), other=0.0)
            b = tl.load(B_ptr + (k + k_range)[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=((k + k_range)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def triton_gemm(A, B, C):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        # strides in elements
        triton_gemm_kernel[( (M + 127)//128, (N + 127)//128 )](
            A.data_ptr(), B.data_ptr(), C.data_ptr(),
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1)
        )

else:
    def triton_elem_add(a, b, out):
        raise RuntimeError("Triton not available")

    def triton_gemm(A, B, C):
        raise RuntimeError("Triton not available")

# 3) CuPy RawKernel (native CUDA) fallback
cuda_elem_add_kernel_source = r'''
extern "C" __global__
void vec_add(const float* x, const float* y, float* z, long N) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = gridDim.x * blockDim.x;
    for (long i = idx; i < N; i += stride) {
        z[i] = x[i] + y[i];
    }
}
'''

cuda_matmul_kernel_source = r'''
extern "C" __global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // simple row-major matmul, one thread per element (inefficient but OK for baseline)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
            acc += A[row*K + k] * B[k*N + col];
        }
        C[row*N + col] = acc;
    }
}
'''

cupy_elem_add = None
cupy_matmul = None
numba_elem_add = None
numba_matmul = None

if HAS_CUPY:
    # compile raw kernels
    cupy_elem_add_module = cupy.RawModule(code=cuda_elem_add_kernel_source, backend='nvcc')
    cupy_elem_add = cupy_elem_add_module.get_function('vec_add')

    cupy_matmul_module = cupy.RawModule(code=cuda_matmul_kernel_source, backend='nvcc')
    cupy_matmul = cupy_matmul_module.get_function('matmul')

    def cupy_elem_add_fn(x, y, z):
        N = x.size
        threads = 256
        blocks = min(1024, (N + threads - 1) // threads)
        cupy_elem_add((blocks,), (threads,), (x, y, z, x.size))
        cupy.cuda.Stream.null.synchronize()

    def cupy_matmul_fn(A, B, C):
        M, K = A.shape
        K2, N = B.shape
        threads = (16, 16)
        blocks = ((N + threads[0]-1)//threads[0], (M + threads[1]-1)//threads[1])
        cupy_matmul((blocks,), (threads,), (A, B, C, M, N, K))
        cupy.cuda.Stream.null.synchronize()

elif HAS_NUMBA:
    # Numba fallback: simple kernels
    @nbcuda.jit
    def numba_vec_add(x, y, z, N):
        i = nbcuda.grid(1)
        stride = nbcuda.gridsize(1)
        while i < N:
            z[i] = x[i] + y[i]
            i += stride

    @nbcuda.jit
    def numba_matmul_kernel(A, B, C, M, N, K):
        row, col = nbcuda.grid(2)
        if row < M and col < N:
            acc = 0.0
            for k in range(K):
                acc += A[row, k] * B[k, col]
            C[row, col] = acc

    def numba_elem_add_fn(x, y, z):
        threads = 256
        blocks = (x.size + threads - 1) // threads
        numba_vec_add[blocks, threads](x, y, z, x.size)
        nbcuda.synchronize()

    def numba_matmul_fn(A, B, C):
        threads = (16, 16)
        blocks = ((A.shape[0] + threads[0]-1)//threads[0], (B.shape[1] + threads[1]-1)//threads[1])
        numba_matmul_kernel[blocks, threads](A, B, C, A.shape[0], B.shape[1], A.shape[1])
        nbcuda.synchronize()
else:
    # no native kernels available
    pass

# ---------- Benchmark runner ----------
def benchmark_elementwise(n_elems=int(1e7), iters=50):
    print("=== Elementwise Add benchmark ===")
    N = int(n_elems)
    # prepare data
    a_t = torch.randn(N, device='cuda', dtype=torch.float32)
    b_t = torch.randn(N, device='cuda', dtype=torch.float32)
    out_t = torch.empty_like(a_t)

    # PyTorch
    t_pytorch = torch_time_fn(lambda: pytorch_elem_add(a_t, b_t, out_t), iters=iters)
    bw_pytorch = (a_t.element_size() * 3 * N) / (t_pytorch/1000.0) / 1e9  # GB/s
    print(f"PyTorch elem_add: {t_pytorch:.3f} ms (avg), approx bandwidth {bw_pytorch:.2f} GB/s")

    # Triton (if)
    if HAS_TRITON:
        t_triton = torch_time_fn(lambda: triton_elem_add(a_t, b_t, out_t), iters=iters)
        bw_triton = (a_t.element_size() * 3 * N) / (t_triton/1000.0) / 1e9
        print(f"Triton elem_add: {t_triton:.3f} ms (avg), approx bandwidth {bw_triton:.2f} GB/s")
    else:
        print("Triton not available, skipping Triton elementwise test.")

    # CuPy / Numba
    if HAS_CUPY:
        xa = cupy.asarray(a_t)
        xb = cupy.asarray(b_t)
        zc = cupy.empty_like(xa)
        t_cupy = cupy_time_fn(lambda: cupy_elem_add(xa, xb, zc), iters=iters)
        bw_cupy = (4 * 3 * N) / (t_cupy/1000.0) / 1e9
        print(f"CuPy RawKernel elem_add: {t_cupy:.3f} ms (avg), approx bandwidth {bw_cupy:.2f} GB/s")
    elif HAS_NUMBA:
        xa = nbcuda.to_device(a_t.cpu().numpy())
        xb = nbcuda.to_device(b_t.cpu().numpy())
        zc = nbcuda.device_array_like(xa)
        t_numba = numba_time_fn(lambda: numba_elem_add_fn(xa, xb, zc), iters=iters)
        bw_numba = (4 * 3 * N) / (t_numba/1000.0) / 1e9
        print(f"Numba cuda elem_add: {t_numba:.3f} ms (avg), approx bandwidth {bw_numba:.2f} GB/s")
    else:
        print("No CuPy/Numba native kernel available, skipping native CUDA test.")

def benchmark_gemm(M=2048, K=2048, N=2048, iters=20):
    print("=== GEMM benchmark ===")
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    C = torch.empty((M, N), device='cuda', dtype=torch.float32)

    # PyTorch (cuBLAS)
    t_pytorch = torch_time_fn(lambda: pytorch_gemm(A, B, C), iters=iters)
    # compute GFLOPS: 2*M*N*K ops
    gflops = (2.0 * M * N * K) / (t_pytorch/1000.0) / 1e9
    print(f"PyTorch matmul: {t_pytorch:.3f} ms (avg), approx {gflops:.1f} GFLOPS")

    # Triton GEMM (if available)
    if HAS_TRITON:
        # Triton kernel expects contiguous tensors
        A_t = A.contiguous()
        B_t = B.contiguous()
        C_t = torch.empty_like(C)
        t_triton = torch_time_fn(lambda: triton_gemm(A_t, B_t, C_t), iters=iters)
        gflops_triton = (2.0 * M * N * K) / (t_triton/1000.0) / 1e9
        print(f"Triton matmul: {t_triton:.3f} ms (avg), approx {gflops_triton:.1f} GFLOPS")
    else:
        print("Triton not available, skipping Triton GEMM test.")

    # CuPy / Numba GEMM (naive kernel; will be slow compared to cuBLAS)
    if HAS_CUPY:
        A_c = cupy.asarray(A)
        B_c = cupy.asarray(B)
        C_c = cupy.empty((M, N), dtype=cupy.float32)
        t_cupy = cupy_time_fn(lambda: cupy_matmul(A_c, B_c, C_c), iters=iters)
        gflops_cupy = (2.0 * M * N * K) / (t_cupy/1000.0) / 1e9
        print(f"CuPy naive matmul kernel: {t_cupy:.3f} ms (avg), approx {gflops_cupy:.1f} GFLOPS (naive kernel, not cuBLAS)")
    elif HAS_NUMBA:
        # create numba device arrays
        A_n = nbcuda.to_device(A.cpu().numpy())
        B_n = nbcuda.to_device(B.cpu().numpy())
        C_n = nbcuda.device_array((M, N), dtype=A.cpu().numpy().dtype)
        t_numba = numba_time_fn(lambda: numba_matmul_fn(A_n, B_n, C_n), iters=iters)
        gflops_numba = (2.0 * M * N * K) / (t_numba/1000.0) / 1e9
        print(f"Numba naive matmul kernel: {t_numba:.3f} ms (avg), approx {gflops_numba:.1f} GFLOPS (naive kernel)")
    else:
        print("No CuPy/Numba native kernel available, skipping native CUDA GEMM test.")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elem-n", type=float, default=1e7, help="number of elements for elementwise test")
    parser.add_argument("--iter-elem", type=int, default=50)
    parser.add_argument("--M", type=int, default=2048)
    parser.add_argument("--K", type=int, default=2048)
    parser.add_argument("--N", type=int, default=2048)
    parser.add_argument("--iter-gemm", type=int, default=20)
    args = parser.parse_args()

    print("Environment:")
    print(f"  PyTorch      : {torch.__version__}")
    print(f"  Triton avail : {HAS_TRITON}")
    print(f"  CuPy avail   : {HAS_CUPY}")
    print(f"  Numba avail  : {HAS_NUMBA}")
    print(f"  Device       : {DEVICE}, current device: {torch.cuda.current_device()}, name: {torch.cuda.get_device_name(0)}")
    print()

    benchmark_elementwise(n_elems=args.elem_n, iters=args.iter_elem)
    print()
    benchmark_gemm(M=args.M, K=args.K, N=args.N, iters=args.iter_gemm)

if __name__ == "__main__":
    main()
