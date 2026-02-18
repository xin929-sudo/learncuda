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
__global__ void findMaxGpu_native(int* data,int n,int *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int local_max = INT_MIN;
    
    for(int i = tid; i < n; i += stride) {
        if(data[i] > local_max) {
            local_max = data[i];
        }
    }
    atomicMax(result,local_max);
}

__global__ void findMaxGpu_shared(int *data,int n,int *result) {
    // 1.分配共享内存
    extern __shared__ int shared_data[];
    
    // 线程索引
    int tid = threadIdx.x; // block内线程索引（0 ~ blockDim.x - 1)
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    
    // 2.每个线程找自己负责数据的最大值
    int local_max = INT_MIN;
    for(int i = gid; i < n; i += stride) {
        if(data[i] > local_max) {
            local_max = data[i];
        }
    }
    
    // 写入共享内存
    shared_data[tid] = local_max;
    // 同步
    __syncthreads();

    // 3.归约树：在block内找最大值
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(tid < offset) {
            // 比较更新
            if(shared_data[tid + offset] > shared_data[tid]) {
                shared_data[tid] = shared_data[tid + offset];
            }
        }
        // 同步！确保当前层的比较都完成了
        __syncthreads();
    }
    // 4.block的代表（线程0）更新全局结果
    if(tid == 0) {
        atomicMax(result,shared_data[0]);
    }
    
}
int main() {

    printf("lesson 3: 统一内存管理求最大值\n");

    // 1.generate dta;
    const int N = 10000000;
    printf("数据大小：  %2.f MB \n\n", N * sizeof(int) / 1024.0 / 1024.0);

    // unified memory
    int* data,* gpu_result,*result_shared;
    CUDA_CHECK(cudaMallocManaged(&data,N * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&gpu_result,sizeof(int)));
     CUDA_CHECK(cudaMallocManaged(&result_shared,sizeof(int)));
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
    
    printf("CPU结果: %d (耗时: %.2f ms)\n\n", cpu_max, cpu_time);


    *gpu_result = INT_MIN;
    int threadsPerBlock = 256;
    // int blockPreGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int blockPreGrid = 1024;

    cudaEvent_t start1,stop1;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));

    cudaEventRecord(start1);

    findMaxGpu_native<<<blockPreGrid,threadsPerBlock>>>(data,N,gpu_result);

    CUDA_CHECK(cudaEventRecord(stop1));

    cudaEventSynchronize(stop1);

    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time,start1,stop1));
    cudaDeviceSynchronize();
    printf("GPU结果: %d (耗时: %.2f ms)\n\n", *gpu_result, gpu_time);
    
    *result_shared = INT_MIN;

    // 共享内存大小（动态申请）
    int shared_mem_size = threadsPerBlock * sizeof(int);
    
    cudaEvent_t start2,stop2;
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));

    cudaEventRecord(start2);

    findMaxGpu_shared<<<blockPreGrid,threadsPerBlock,shared_mem_size>>>(data,N,result_shared);

    CUDA_CHECK(cudaEventRecord(stop2));

    cudaEventSynchronize(stop2);

    float time_shared;
    CUDA_CHECK(cudaEventElapsedTime(&time_shared,start2,stop2));
    cudaDeviceSynchronize();
    printf("gpu_shared结果: %d (耗时: %.2f ms)\n\n", *result_shared, time_shared);


    // 内存需要释放掉
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(gpu_result));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaFree(result_shared));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));
    
}