#include<stdio.h>
#include<cuda_runtime.h>
#include<limits.h>
#include<cooperative_groups.h>

namespace cg = cooperative_groups;
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

// cpu version matrix matmul 
// A is (M,K), B is (K,N)
void matmulCPU(float* A,float* B,float* C,int M,int N,int K) {
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for(int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
                    // A[i][k] * B[k][j]
            }
            C[i * N + j] = sum;
        }
    }
}

// gpu version :navive matmul
__global__ void matmul_native(float* A,float* B,float* C,int M,int N,int K) {

    // 计算线程负责的C元素位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if(row < M && col < N) {
        float sum = 0.0f;

        // 计算点积：A的row 行，乘以 B 的 col 列
        for(int k = 0; k < K; ++k){
            sum += A[row * K + k] * B[k * N + col];
                // A[row][k] * B[k][col]
        }
            // 写回结果
        C[row * N + col] = sum; 
    }
}
// gpu version: tile matmul
#define TILE_SIZE 16
__global__ void matmalGPU_tiled(float* A,float* B,float* C,int M,int N,int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // 定义shared memory
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE; // 进行划分，因为一次装不了这么多数据

    for(int t = 0; t < numTiles; ++t){
        // load A B to shared memory
         int aCol = t * TILE_SIZE + tx;
         if(row < M && aCol < K) {
            As[ty][tx] = A[row * K + aCol];
         } else {
            As[ty][tx] = 0.0f;
         }

         int bRow = t * TILE_SIZE + ty;
         if(col < N && bRow < K) {
            Bs[ty][tx] = B[bRow * N + col];
         } else {
            Bs[ty][tx] = 0.0f;
         }
         // 等待所有数据
         __syncthreads();
         // 使用 shared memory  to calculate tiles value
         for(int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
         }

         __syncthreads();
    }

    // write back
    if(row < M && col < N) {
        C[row * N + col] = sum;
    }
}
// 每个线程需要计算 4*4 个数据
__global__ void matmalGPU_tiled4(float* A,float* B,float* C,int M,int N,int K) {
    
    cg::thread_block block = cg::this_thread_block();

    // 计算 当前线程块负责的C矩阵的起始位置
    int blockRow = blockIdx.y * TILE_SIZE * 4;
    int blockCol = blockIdx.x * TILE_SIZE * 4;
    // 计算 当前线程在线程块内的坐标
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    // 定义shared memory，注意这里的尺寸需要调整，因为每个线程要处理4*4的数据
    // 根据 C = A * B 的计算方式，A的tile需要是 64*K，B的tile需要是 K*64，
    // A矩阵需要提供64行数据，B矩阵需要提供64列数据
    // K的维度我们一次装不下，所以需要分块加载
    __shared__ float As[TILE_SIZE * 4][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE * 4];
   // 进行划分，因为一次装不了这么多数据
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE; // 向上取整
    float sum[4][4] = {{0.0f}};
    for(int t = 0; t < numTiles; ++t){
        // load A B to shared memory
        // tile_k 是当前块加载的A和B的起始列/
        // int tile_k = t * numTiles;

        for(int i = 0; i < 4; ++i) {
                        // 大块的起始行，当前线程要处理的4个数据，i是偏移量
            int aRow = blockRow + ty * 4 + i; // A的行跟 C的行有关
            int aCol = t*TILE_SIZE + tx;      // A的列跟 K有关，而跟C的行无关     
            As[4 * ty + i][tx] = (aRow < M && aCol < K) ? (A[aRow * K + aCol]) : 0.0f;
                        // tile_k对应A矩阵的列
           
        }
        for(int j = 0; j < 4; ++j) {
            // Bs 是 16 * 64 
                    // 大块的起始行，当前线程要处理的4个数据，i是偏移量
            int bRow = t * TILE_SIZE + ty; // B 的行 跟 C无关，只跟 K 有关
            int bCol = blockCol + tx * 4 + j; // B的列 跟 C有关
                    // tile_k对应A矩阵的列
            Bs[ty][4 * tx + j] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
            // As 是 64 * 16，一个block里面 有 16 * 16 个线程，列好处理
            
        }
        // 等待线程块读取
        block.sync();
        
        for(int k = 0; k < TILE_SIZE; ++k) {
            float a_reg[4],b_reg[4];
            // 每个线程需要计算4*4个数据，所以需要把A的4行和B的4列加载到寄存器
            for(int i = 0; i < 4; ++i) {
                a_reg[i] = As[ty *4 + i][k];
            }
            for(int j = 0; j < 4; ++j) {
                b_reg[j] = Bs[k][tx * 4 + j];
            }
            for(int i = 0; i < 4; ++i){
                for(int j = 0; j < 4 ;j++){
                    sum[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        block.sync();
    }
    // write back
    for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 4; ++j) {
            // 计算当前线程负责的C元素位置
            // 大块的起始行，当前线程要处理的4个数据，i是偏移量
            // 大块的起始列，当前线程要处理的4个数据，j是偏移量
            // 注意边界检查，因为矩阵维度不一定是64的倍数
           int c_row = blockRow + ty * 4 + i;
           int c_col = blockCol + tx * 4 + j;

           if(c_row < M && c_col < N) {
                C[c_row * N + c_col] = sum[i][j];
           }
        }
    }
}
void initMatrixRandom(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;  // 0-1之间的随机数
    }
}
// 验证结果
bool verifyResult(float* C_cpu,float* C_gpu,int M,int N){
    const float epsilon = 1e-3;
    int errorCount = 0;
    
    for(int i = 0; i < M * N; ++i) {
        float diff = fabs(C_cpu[i] - C_gpu[i]);
        if(diff > epsilon) {
            errorCount++;

            if(errorCount <= 10) {
                printf("错误[%d]: CPU = %.6f, GPU = %.6f, diff = %.6f\n",
                i,C_cpu[i],C_gpu[i],diff);
            }
        }
    }
    if(errorCount > 0) {
        printf("发现 %d 个错误 (总共 %d 个元素)\n",errorCount, M * N);
        return false;
    }
    return true;
}
int main() {

    printf("lesson 4: 统一内存管理求矩阵乘法\n");

    // 1.generate dta;
    int M = 1024;
    int K = 1024;
    int N = 1024;
    
    printf("矩阵维度：\n");
    printf("    A: %d x %d\n",M,K);
    printf("    B: %d x %d\n",K,N);
    printf("    C: %d x %d\n",M,N);

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    printf("内存占用:\n");
    printf("    A:%.2f MB\n",size_A / 1024.0f / 1024.0f);
    printf("    B:%.2f MB\n",size_B / 1024.0f / 1024.0f);
    printf("    C:%.2f MB\n",size_C / 1024.0f / 1024.0f);
    printf("    总计:%.2f MB\n",(size_A + size_B + size_C)/ 1024.0f / 1024.0f);

    // 分配Host内存
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_cpu = (float*)malloc(size_C);
    float *h_C_gpu = (float*)malloc(size_C);
    float *h_C_gpu_tiled = (float*)malloc(size_C);
     float *h_C_gpu_tiled4 = (float*)malloc(size_C);
    if(!h_A || !h_B || !h_C_cpu || !h_C_gpu || !h_C_gpu_tiled|| !h_C_gpu_tiled4) {
        fprintf(stderr,"Host 内存分配失败!\n");
        exit(1);
    }

    // 分配Device内存
    float *d_A ,*d_B,*d_C_gpu,*d_C_gpu_tiled,*d_C_gpu_tiled4;
    CUDA_CHECK(cudaMalloc(&d_A,size_A));
    CUDA_CHECK(cudaMalloc(&d_B,size_B));
    CUDA_CHECK(cudaMalloc(&d_C_gpu_tiled,size_C));
    CUDA_CHECK(cudaMalloc(&d_C_gpu,size_C));
    CUDA_CHECK(cudaMalloc(&d_C_gpu_tiled4,size_C));
    // 初始化数据 （在 Host 上)
    printf("初始化数据 （在 Host 上)...\n");
    srand(time(NULL));
    initMatrixRandom(h_A,M,K);
    initMatrixRandom(h_B,K,N);

    printf("拷贝数据到 Device\n");
    CUDA_CHECK(cudaMemcpy(d_A,h_A,size_A,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B,h_B,size_B,cudaMemcpyHostToDevice));




    clock_t cpu_start = clock();
    matmulCPU(h_A, h_B, h_C_cpu, M, N, K);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000;

    // 计算GLOPS
    // 矩阵乘法的浮点操作数：每个输出元素需要K次乘法+K次加法 = 2K次操作
    // 总操作数： M * N * 2K

    double gflops_cpu = (2.0 * M * N * K) / (cpu_time / 1000.0) / 1e9;
    
    printf("CPU完成！\n");
    printf("  耗时: %.2f ms\n", cpu_time);
    printf("  性能: %.2f GFLOPS\n", gflops_cpu);
    printf("  示例结果: C[0][0]=%.2f, C[10][10]=%.2f\n\n",
           h_C_cpu[0], h_C_cpu[10 * N + 10]);
        
    // 配置2D线程块和网络
    dim3 blockDim(16,16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,(M + blockDim.y - 1) / blockDim.y);
    printf("Kernel配置：\n");
    printf("    Block: (%d,%d) = %d thread\n",blockDim.x,blockDim.y,blockDim.x * blockDim.y);
    printf("    Grid: (%d,%d) = %d block\n",gridDim.x,gridDim.y,gridDim.x * gridDim.y);
               

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
    matmul_native<<<gridDim,blockDim>>>(d_A,d_B,d_C_gpu,M,N,K);
    
    // 2. 必须同步！等待搬运和第一次计算彻底结束
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("预热完成！数据已在显存中，开始真正的性能测试...\n\n");
    // =======================================================

    printf("========================================\n");
    printf("GPU计算（朴素实现）\n");
    printf("========================================\n");

    cudaEvent_t start1,stop1;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));

    cudaEventRecord(start1);

    matmul_native<<<gridDim,blockDim>>>(d_A,d_B,d_C_gpu,M,N,K);

    CUDA_CHECK(cudaEventRecord(stop1));

    cudaEventSynchronize(stop1);

    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time,start1,stop1));
    cudaDeviceSynchronize();
    // printf("GPU 结果: %d (耗时: %.2f ms)\n\n", *gpu_result, gpu_time);
    
    double gflops_gpu_naive = (2.0 * M * N * K) / (gpu_time / 1000.0) / 1e9;
    // 拷贝结果会Host
    CUDA_CHECK(cudaMemcpy(h_C_gpu,d_C_gpu,size_C,cudaMemcpyDeviceToHost));
    printf("GPU-naive完成！\n");
    printf("  耗时: %.2f ms\n", gpu_time);
    printf("  性能: %.2f GFLOPS\n", gflops_gpu_naive);
    printf("  示例结果: C[0][0]=%.2f, C[10][10]=%.2f\n\n",
           h_C_gpu[0], h_C_gpu[10 * N + 10]);
   
    printf("========================================\n");
    printf("验证结果...\n");
    if (verifyResult(h_C_cpu, h_C_gpu, M, N)) {
        printf("✓ 结果正确！GPU计算与CPU一致\n");
    } else {
        printf("✗ 结果错误！请检查代码\n");
    }
    
    printf("========================================\n");
    printf("GPU计算（共享内存和分块矩阵）\n");
    printf("========================================\n");


    cudaEvent_t start2,stop2;
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));

    cudaEventRecord(start2);

    matmalGPU_tiled<<<gridDim,blockDim>>>(d_A,d_B,d_C_gpu_tiled,M,N,K);

    CUDA_CHECK(cudaEventRecord(stop2));

    cudaEventSynchronize(stop2);

    float gpu_time_tiled;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_tiled,start2,stop2));
    cudaDeviceSynchronize();
    
    double gflops_gpu_tiled = (2.0 * M * N * K) / (gpu_time_tiled / 1000.0) / 1e9;
    // 拷贝结果会Host
    CUDA_CHECK(cudaMemcpy(h_C_gpu_tiled,d_C_gpu_tiled,size_C,cudaMemcpyDeviceToHost));
    printf("GPU-tiled完成！\n");
    printf("  耗时: %.2f ms\n", gpu_time_tiled);
    printf("  性能: %.2f GFLOPS\n", gflops_gpu_tiled);
    printf("  示例结果: C[0][0]=%.2f, C[10][10]=%.2f\n\n",
           h_C_gpu_tiled[0], h_C_gpu_tiled[10 * N + 10]);
   
    printf("========================================\n");
    printf("验证结果...\n");
    if (verifyResult(h_C_cpu, h_C_gpu_tiled, M, N)) {
        printf("✓ 结果正确！GPU计算与CPU一致\n");
    } else {
        printf("✗ 结果错误！请检查代码\n");
    }

    printf("========================================\n");
    printf("GPU计算（共享内存和分块矩阵（4 * 4）\n");
    printf("========================================\n");

    dim3 blockDim_cg(16, 16); // 每个线程处理 4*4 小块，总共就是 64*64个小块
    dim3 gridDim_cg((N + 64 - 1) / 64,
                 (M + 64 - 1) / 64);
    cudaEvent_t start3,stop3;
    CUDA_CHECK(cudaEventCreate(&start3));
    CUDA_CHECK(cudaEventCreate(&stop3));

    cudaEventRecord(start3);

    matmalGPU_tiled4<<<gridDim_cg,blockDim_cg>>>(d_A,d_B,d_C_gpu_tiled4,M,N,K);

    CUDA_CHECK(cudaEventRecord(stop3));

    cudaEventSynchronize(stop3);

    float gpu_time_tiled4;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_tiled4,start3,stop3));
    cudaDeviceSynchronize();
    
    double gflops_gpu_tiled4 = (2.0 * M * N * K) / (gpu_time_tiled4 / 1000.0) / 1e9;
    // 拷贝结果会Host
    CUDA_CHECK(cudaMemcpy(h_C_gpu_tiled4,d_C_gpu_tiled4,size_C,cudaMemcpyDeviceToHost));
    printf("GPU-tiled完成！\n");
    printf("  耗时: %.2f ms\n", gpu_time_tiled4);
    printf("  性能: %.2f GFLOPS\n", gflops_gpu_tiled4);
    printf("  示例结果: C[0][0]=%.2f, C[10][10]=%.2f\n\n",
           h_C_gpu_tiled4[0], h_C_gpu_tiled4[10 * N + 10]);
   
    printf("========================================\n");
    printf("验证结果...\n");
    if (verifyResult(h_C_cpu, h_C_gpu_tiled4, M, N)) {
        printf("✓ 结果正确！GPU计算与CPU一致\n");
    } else {
        printf("✗ 结果错误！请检查代码\n");
    }
    // 内存需要释放掉
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    free(h_C_gpu_tiled);
    free(h_C_gpu_tiled4);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_gpu));
    CUDA_CHECK(cudaFree(d_C_gpu_tiled));
    CUDA_CHECK(cudaFree(d_C_gpu_tiled4));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));
    CUDA_CHECK(cudaEventDestroy(start3));
    CUDA_CHECK(cudaEventDestroy(stop3));
    
}