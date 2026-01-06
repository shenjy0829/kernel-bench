// ...existing code...
#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << "\n\n";

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties for device " << i << " failed: " << cudaGetErrorString(err) << "\n";
            continue;
        }

        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total global memory: " << (prop.totalGlobalMem >> 20) << " MB\n";
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "  Shared memory per SM: " << prop.sharedMemPerMultiprocessor << " bytes\n";
        std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
        std::cout << "  Registers per SM: " << prop.regsPerMultiprocessor << " registers\n";
        std::cout << "  Warp size: " << prop.warpSize << "\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  Number of SMs: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << "\n";
        std::cout << "  Max grid dimensions: ["
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << "]\n";
        std::cout << "  Max threads dim (block): ["
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << "]\n";

        // Use cudaDeviceGetAttribute for clock and memory clock (more portable across CUDA versions)
        int clockRateKHz = 0;
        if (cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, i) == cudaSuccess) {
            std::cout << "  Clock rate: " << (clockRateKHz / 1000) << " MHz\n";
        } else {
            std::cout << "  Clock rate: N/A\n";
        }

        int memClockKHz = 0;
        if (cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, i) == cudaSuccess) {
            std::cout << "  Memory Clock Rate: " << (memClockKHz / 1000) << " MHz\n";
        } else {
            std::cout << "  Memory Clock Rate: N/A\n";
        }

        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
        std::cout << "  L2 Cache Size: " << prop.l2CacheSize << " bytes\n";
        std::cout << "  PCI Domain/Bus/Device: "
                  << prop.pciDomainID << "/" << prop.pciBusID << "/" << prop.pciDeviceID << "\n";
        std::cout << std::endl;
    }

    return 0;
}
// ...existing code...