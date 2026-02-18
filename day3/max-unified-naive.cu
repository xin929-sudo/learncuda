#include<stdio.h>
#include<cuda_runtime.h>
#include<limits.h>

// 宏定义：包裹函数调用，检查返回值
#define CUDA_CHECK(call) \
do { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while (0)

// cpu version find max
int findMaxCpu(int* data,int n) {
    int max_val = INT_MIN;
    for(int i = 0 ; i < n; ++i) {
        if(data[i] > max_val) {
            max_val = data[i];
        }
    }
    return max_val;
}

// gpu version :navive cal max
__global__ void findMaxGpu_native(int* data,int n,int *reslut) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int local_max = INT_MIN;
    
    for(int i = tid; i < n; i += stride) {
        if(data[i] > local_max) {
            local_max = data[i];
        }
    }
    atomicMax(reslut,local_max);
}
int main() {

    printf("lesson 3: 统一内存管理求最大值\n");

    // 1.generate dta;
    const int N = 10000000;
    printf("数据大小：  %2.f MB \n\n", N * sizeof(int) / 1024.0 / 1024.0);

    // unified memory
    int* data,* gpu_result;
    CUDA_CHECK(cudaMallocManaged(&data,N * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&gpu_result,N * sizeof(int)));

    srand(time(NULL));
    for(int i = 0; i < N; i++) {
        data[i] = rand() % 100000; // 0 - 99999
    }

    // 设定一个最大值
    int know_max_pos = N / 2;
    data[know_max_pos] = 999999;

    printf("记住啦 ： 最大值是 999999\n\n");

    clock_t cpu_start = clock();
    int cpu_max = findMaxCpu(data,N);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;
    printf("cpu 耗时 %0.2f ms \n\n",cpu_time);


    *gpu_result = INT_MIN;
    int threadsPerBlock = 256;
    int blockPreGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // int blockPreGrid = 1024;

    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaEventRecord(start);

    findMaxGpu_native<<<blockPreGrid,threadsPerBlock>>>(data,N,gpu_result);

    CUDA_CHECK(cudaEventRecord(stop));

    cudaEventSynchronize(stop);

    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time,start,stop));
    cudaDeviceSynchronize();

    printf("gpu 耗时 %0.2f ms \n\n",gpu_time);

    // 内存需要释放掉
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(gpu_result));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
}