#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// 一维网格 + 一维 block 的矩阵加法示例
// 每个线程处理一个矩阵元素（按行优先存储）

__global__ void mat_add_kernel(const float* A, const float* B, float* C, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char** argv) {
    int N = 1024; // 默认矩阵维度 N x N
    if (argc > 1) N = atoi(argv[1]);
    int total = N * N;
    size_t bytes = (size_t)total * sizeof(float);

    std::cout << "Matrix add test (" << N << " x " << N << ") using 1D grid & 1D block\n";

    // host
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        std::cerr << "Host allocation failed\n";
        return 1;
    }

    // init
    for (int i = 0; i < total; ++i) {
        h_A[i] = 1.0f * (i % 97);
        h_B[i] = 2.0f * (i % 53);
    }

    // device
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaError_t err;
    err = cudaMalloc((void**)&d_A, bytes); if (err != cudaSuccess) { std::cerr<<"cudaMalloc A failed: "<<cudaGetErrorString(err)<<"\n"; return 1; }
    err = cudaMalloc((void**)&d_B, bytes); if (err != cudaSuccess) { std::cerr<<"cudaMalloc B failed: "<<cudaGetErrorString(err)<<"\n"; return 1; }
    err = cudaMalloc((void**)&d_C, bytes); if (err != cudaSuccess) { std::cerr<<"cudaMalloc C failed: "<<cudaGetErrorString(err)<<"\n"; return 1; }

    err = cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { std::cerr<<"cudaMemcpy A failed: "<<cudaGetErrorString(err)<<"\n"; return 1; }
    err = cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) { std::cerr<<"cudaMemcpy B failed: "<<cudaGetErrorString(err)<<"\n"; return 1; }

    // launch kernel with 1D grid and 1D block
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    // 使用 CUDA 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mat_add_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, total);
    cudaEventRecord(stop);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // copy back
    err = cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost); if (err != cudaSuccess) { std::cerr<<"cudaMemcpy C failed: "<<cudaGetErrorString(err)<<"\n"; return 1; }

    // verify
    size_t errors = 0;
    for (int i = 0; i < total; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            if (errors < 10) std::cerr << "mismatch at " << i << ": got " << h_C[i] << ", expected " << expected << "\n";
            ++errors;
        }
    }

    std::cout << "Kernel time: " << ms << " ms\n";
    if (errors == 0) std::cout << "Result: PASS\n";
    else std::cout << "Result: FAIL (" << errors << " mismatches)\n";

    // cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return (errors == 0) ? 0 : 2;
}
