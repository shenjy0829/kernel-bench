# kernel-bench
For self-learning purposes ~

+ Implementation
    + Pytorch
    + CUDA
    + Cute DSL
    + triton
    + tilelang
+ to do kernel
    + Reduction
    + Prefix Sum
    + Top K Selection
    + K-Means Clustering

    + Elementwise
        + ???
    + GEMM
        + GEMM
        + SGEMM
        + 
    + Attention
        + flash-attention v1
        + flash-attention v2
        + flash-attention v3
        + flash-attention v4
        + Multi-Head Attention
    + Multi-Agent Simulation
    + LDPC
    + FFT
+ Done kernel
    + 

# Usage
## Setup Env
```bash
mamba create --name kernel_bench python=3.11

## cuda toolkit and dsl
# cuda
mamba install cuda-nvcc
# torch & triton
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130   
#  Cute DSL
pip install nvidia-cutlass-dsl


## else
mamba install colorama

## install LazyGPU
cd utils
pip install -e .

```

# Performance
+ NVIDIA GeForce RTX 4090

+ NVIDIA A100-SXM4-40GB



# References
The implementation of this benchmark has benefited from the following sources:
+ [1] [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations on GPUs](https://openreview.net/pdf?id=Y7U8s2pVtX)
