import torch
import triton
import triton.language as tl

@triton.jit
def histogram_kernel(
    input_ptr,          # pointer to the input data
    histogram_ptr,      # pointer to the histogram
    N,                  # number of elements in input
    num_bins,           # number of bins in histogram
    BLOCK_SIZE_N: tl.constexpr,
    NUM_BINS: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets_block = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_block = offsets_block < N
    
    input_block = tl.load(
        input_ptr + offsets_block,
        mask = mask_block,
        other = 0
    )  # [BLOCK_SIZE_N]

    ## why not work？
    h = tl.histogram(
        input_block,
        num_bins=NUM_BINS,
        mask = mask_block
    )  # [NUM_BINS]
    tl.atomic_add(histogram_ptr + tl.arange(0, NUM_BINS), h, mask = h > 0)

    # h = tl.histogram(
    #     input_block,
    #     num_bins=NUM_BINS,
    # )  # [NUM_BINS]
    # offs_numbins = tl.arange(0, NUM_BINS)
    # h -=  (offs_numbins == 0) * (BLOCK_SIZE_N - tl.sum(mask_block.to(tl.int32)))
    # mask_store = (offs_numbins < num_bins) & (h > 0)
    # tl.atomic_add(histogram_ptr + offs_numbins, h, mask=mask_store)


# input, histogram are tensors on the GPU
def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    BLOCK_SIZE_N = 4096
    grid = (triton.cdiv(N, BLOCK_SIZE_N),)
    
    histogram_kernel[grid](
        input,
        histogram,
        N,
        num_bins,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_BINS=triton.next_power_of_2(num_bins)
    )