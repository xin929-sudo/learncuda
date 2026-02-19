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

#define WARP_SIZE   32
__global__ void findMaxGpu_warp_shuffle(int *data,int n,int *result) {

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 计算warp相关信息
    int lane = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;

    // step1:每个线程找到自己负责数据的最大值
    int local_max = INT_MIN;
    for(int i = gid; i < n; i += stride) {
        if(data[i] > local_max) {
            local_max = data[i];
        }
    }


    // step2 : warp shuffle reduce(warp内归约)
    // 利用shuffle指令，在寄存器层面交换数据无需共享内存
    // 第 1 轮：offset = 16
    // 代码执行：neighbor = __shfl_down_sync(..., local_max, 16)

    // 发生了什么：

    // 0 号 拿到 16 号 的数据，两者比大小，留下大的。

    // 1 号 拿到 17 号 的数据，两者比大小。

    // ...

    // 15 号 拿到 31 号 的数据，两者比大小。

    // 结果：此时，全组最大的数，一定已经跑到**前 16 个人（0~15 号）**的手里了。后 16 个人手里虽然也有数据，但已经是“废牌”了。
    // 第 2 轮：offset = 8 (即 16 >> 1)
    // 代码执行：neighbor = __shfl_down_sync(..., local_max, 8)

    // 发生了什么：

    // 0 号 拿到 8 号 的数据（注意：8 号此时手里已经是原本 8 号和 24 号的胜者了）。

    // 1 号 拿到 9 号 的数据。

    // ...

    // 7 号 拿到 15 号 的数据。

    // 结果：全组最大的数，缩小到了**前 8 个人（0~7 号）**的手里。
    for(int offset = 16; offset > 0; offset >>= 1) {
        int neighbnor = __shfl_down_sync(0xffffffff,local_max,offset);
        local_max = max(local_max,neighbnor);
    }
    // step3 :collect all warp results to shared memory
    // 假设 Block 大小为 256，则有 8 个 Warp。
    // 每个 Warp 的 0 号线程(lane==0)持有该 Warp 的最大值，将其写入共享内存。
    __shared__ int warp_maxes[8];
    if(lane == 0) {
        warp_maxes[warpId] = local_max;
    }
    __syncthreads();

    // step4: the laset warp warp does the reduce for all warp_maxes
    // 让第一个warp（warpID = 0)把共享内存的8个值取出来，再做一次归约
    // 因为8个数据分配到不同的线程里面，所以需要统一在一起
    int block_max = INT_MIN;
    if(warpId == 0) {
        if(lane < 8) { // 前8个线程去拿东西，其余空转
            block_max = warp_maxes[lane];
        } 
        // 再次使用 Shuffle 进行归约
        for(int offset = 16; offset > 0; offset >>= 1) {
            int neighbor = __shfl_down_sync(0xffffffff,block_max,offset);
            block_max = max(block_max,neighbor);
        }
    }
    // step5: thread 0 update the global max
    // 此时 lane 0 (也就是 tid 0) 持有整个 Block 的最大值
    if (tid == 0) {
        atomicMax(result, block_max);
    }

}
int main() {

    printf("lesson 3: 统一内存管理求最大值\n");

    // 1.generate dta;
    const int N = 200000000;
    printf("数据大小：  %2.f MB \n\n", N * sizeof(int) / 1024.0 / 1024.0);

    // unified memory
    int* data,* gpu_result,*result_shared,*result_warp;
    CUDA_CHECK(cudaMallocManaged(&data,N * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&gpu_result,sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&result_shared,sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&result_warp,sizeof(int)));
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


    
    // 3. 必须同步！确保上面两步彻底做完了，再开始计时
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("预热完成！开始真正的性能测试...\n\n");

    *gpu_result = INT_MIN;
    int threadsPerBlock = 256;
    // int blockPreGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int blockPreGrid = 1024;

// =======================================================
    // 🚀【修正版】WSL2 专用预热方案
    // =======================================================
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    printf("Debug: Current Device ID = %d (WSL2 环境忽略 Prefetch)\n", deviceId);

    printf("正在预热 GPU... (利用 Kernel 触发缺页中断搬运数据)\n");

    // 1. 【关键】不要用 cudaMemPrefetchAsync，直接跑一次 Kernel
    // 当 GPU 试图读取 data[i] 时，驱动会自动把数据从 CPU 搬到 GPU L2/显存
    // 这次运行会很慢（因为包含搬运时间），我们不计入成绩
    findMaxGpu_warp_shuffle<<<blockPreGrid, threadsPerBlock>>>(data, N, gpu_result);
    
    // 2. 必须同步！等待搬运和第一次计算彻底结束
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("预热完成！数据已在显存中，开始真正的性能测试...\n\n");
    // =======================================================


    // --- 下面是原本的计时代码 (保持不变) ---
    // 此时数据已经在显存里了，第二次运行就会飞快！
   
    
    // 2. 预热 Kernel (消除第一次启动的开销)
    // 随便跑一次，让 GPU 醒过来
    findMaxGpu_warp_shuffle<<<blockPreGrid, threadsPerBlock>>>(data, N, gpu_result);

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
    printf("GPU 结果: %d (耗时: %.2f ms)\n\n", *gpu_result, gpu_time);
    
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
    printf("gpu_shared 结果: %d (耗时: %.2f ms)\n\n", *result_shared, time_shared);

    cudaEvent_t start3,stop3;
    CUDA_CHECK(cudaEventCreate(&start3));
    CUDA_CHECK(cudaEventCreate(&stop3));

    cudaEventRecord(start3);

    findMaxGpu_warp_shuffle<<<blockPreGrid,threadsPerBlock>>>(data,N,result_warp);

    CUDA_CHECK(cudaEventRecord(stop3));

    cudaEventSynchronize(stop3);

    float time_warp;
    CUDA_CHECK(cudaEventElapsedTime(&time_warp,start3,stop3));
    cudaDeviceSynchronize();
    printf("gpu_warp 结果: %d (耗时: %.2f ms)\n\n", *result_warp, time_warp);
    // 内存需要释放掉
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(gpu_result));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaFree(result_shared));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));
    CUDA_CHECK(cudaFree(result_warp));
    CUDA_CHECK(cudaEventDestroy(start3));
    CUDA_CHECK(cudaEventDestroy(stop3));
    
}