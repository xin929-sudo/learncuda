// 简单的 CUDA 程序：输出显卡信息
#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Detected " << deviceCount << " CUDA device(s)\n";

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed for device " << i << ": " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
        // Some CUDA headers/platforms may not expose `prop.clockRate` member.
        // Use cudaDeviceGetAttribute as a portable alternative (returns kHz).
        int clockKHz = 0;
        err = cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, i);
        if (err == cudaSuccess) {
            std::cout << "  Clock rate: " << (clockKHz / 1000.0) << " MHz" << std::endl;
        } else {
            std::cout << "  Clock rate: (unknown)" << std::endl;
        }
        std::cout << "  PCI Bus ID: " << prop.pciBusID << ", PCI Device ID: " << prop.pciDeviceID << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
