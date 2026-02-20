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

// gpu version :navive cal max
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

    // unified memory
    float *A,*B,*C_cpu,*C_gpu;
    CUDA_CHECK(cudaMallocManaged(&A,size_A));
    CUDA_CHECK(cudaMallocManaged(&B,size_B));
    CUDA_CHECK(cudaMallocManaged(&C_cpu,size_C));
    CUDA_CHECK(cudaMallocManaged(&C_gpu,size_C));

    printf("初始化矩阵...\n");
    srand(time(NULL));
    initMatrixRandom(A,M,K);
    initMatrixRandom(B,K,N);

    clock_t cpu_start = clock();
    matmulCPU(A, B, C_cpu, M, N, K);
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
           C_cpu[0], C_cpu[10 * N + 10]);


    

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
    matmul_native<<<gridDim,blockDim>>>(A,B,C_gpu,M,N,K);
    
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

    matmul_native<<<gridDim,blockDim>>>(A,B,C_gpu,M,N,K);

    CUDA_CHECK(cudaEventRecord(stop1));

    cudaEventSynchronize(stop1);

    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time,start1,stop1));
    cudaDeviceSynchronize();
    // printf("GPU 结果: %d (耗时: %.2f ms)\n\n", *gpu_result, gpu_time);
    
    double gflops_gpu_naive = (2.0 * M * N * K) / (gpu_time / 1000.0) / 1e9;
    printf("GPU-naive完成！\n");
    printf("  耗时: %.2f ms\n", gpu_time);
    printf("  性能: %.2f GFLOPS\n", gflops_gpu_naive);
    printf("  示例结果: C[0][0]=%.2f, C[10][10]=%.2f\n\n",
           C_gpu[0], C_gpu[10 * N + 10]);
   
    printf("========================================\n");
    printf("验证结果...\n");
    if (verifyResult(C_cpu, C_gpu, M, N)) {
        printf("✓ 结果正确！GPU计算与CPU一致\n");
    } else {
        printf("✗ 结果错误！请检查代码\n");
    }

    // 内存需要释放掉
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C_cpu));
    CUDA_CHECK(cudaFree(C_gpu));
    // CUDA_CHECK(cudaFree(gpu_result));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));


    
}