#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Timing utilities
typedef struct {
    struct timespec start;
    struct timespec end;
} Timer;

void timer_start(Timer *timer) {
    clock_gettime(CLOCK_MONOTONIC, &timer->start);
}

double timer_end(Timer *timer) {
    clock_gettime(CLOCK_MONOTONIC, &timer->end);
    double elapsed = (timer->end.tv_sec - timer->start.tv_sec) + 
                     (timer->end.tv_nsec - timer->start.tv_nsec) / 1e9;
    return elapsed;
}

// Matrix initialization
void init_matrix_zero(float *matrix, int rows, int cols) {
    memset(matrix, 0, rows * cols * sizeof(float));
}

void init_matrix_random(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

// CPU baseline kernel
void matmul_cpu_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float tmp = 0.0f;
            for (int k = 0; k < K; k++) {
                tmp += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = tmp;
        }
    }
}

// ============================================================================
// GPU KERNELS
// ============================================================================

// Naive GPU kernel: one thread per output element
__global__ void matmul_gpu_naive(const float *A, const float *B, float *C, 
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float tmp = 0.0f;
        for (int k = 0; k < K; k++) {
            tmp += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = tmp;
    }
}

// ============================================================================
// KERNEL LAUNCHER FUNCTIONS
// ============================================================================

void launch_matmul_gpu_naive(const float *d_A, const float *d_B, float *d_C,
                              int M, int N, int K) {
    dim3 blockDim(16, 16);  // 16x16 = 256 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    
    matmul_gpu_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// BENCHMARKING INFRASTRUCTURE
// ============================================================================

// Statistics computation
typedef struct {
    double min;
    double max;
    double avg;
    double median;
} Stats;

int compare_doubles(const void *a, const void *b) {
    double diff = (*(double*)a - *(double*)b);
    return (diff > 0) - (diff < 0);
}

Stats compute_stats(double *times, int n_runs) {
    Stats stats;
    stats.min = DBL_MAX;
    stats.max = -DBL_MAX;
    stats.avg = 0.0;
    
    double *sorted_times = (double*)malloc(n_runs * sizeof(double));
    memcpy(sorted_times, times, n_runs * sizeof(double));
    qsort(sorted_times, n_runs, sizeof(double), compare_doubles);
    
    for (int i = 0; i < n_runs; i++) {
        if (times[i] < stats.min) stats.min = times[i];
        if (times[i] > stats.max) stats.max = times[i];
        stats.avg += times[i];
    }
    stats.avg /= n_runs;
    
    if (n_runs % 2 == 0) {
        stats.median = (sorted_times[n_runs/2 - 1] + sorted_times[n_runs/2]) / 2.0;
    } else {
        stats.median = sorted_times[n_runs/2];
    }
    
    free(sorted_times);
    return stats;
}

void print_stats(const char *kernel_name, Stats stats, int M, int N, int K) {
    double flops = 2.0 * M * N * K;
    
    printf("\n%s Statistics:\n", kernel_name);
    printf("  Min:    %.6f seconds (%.2f GFLOPS)\n", 
           stats.min, flops / (stats.min * 1e9));
    printf("  Max:    %.6f seconds (%.2f GFLOPS)\n", 
           stats.max, flops / (stats.max * 1e9));
    printf("  Avg:    %.6f seconds (%.2f GFLOPS)\n", 
           stats.avg, flops / (stats.avg * 1e9));
    printf("  Median: %.6f seconds (%.2f GFLOPS)\n", 
           stats.median, flops / (stats.median * 1e9));
    printf("\n");
    fflush(stdout);
}

void print_usage(const char *prog_name) {
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("Options:\n");
    printf("  -m M          Set M dimension (default: 4096)\n");
    printf("  -n N          Set N dimension (default: 4096)\n");
    printf("  -k K          Set K dimension (default: 4096)\n");
    printf("  -z            Use zero initialization (default: random)\n");
    printf("  -r RUNS       Number of runs for benchmarking (default: 5)\n");
    printf("  -h            Show this help message\n");
}

int main(int argc, char **argv) {
    // Default parameters
    int M = 4096;
    int N = 4096;
    int K = 4096;
    bool use_zero_init = false;
    int n_runs = 5;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0) {
            use_zero_init = true;
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            n_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate parameters
    if (M <= 0 || N <= 0 || K <= 0) {
        printf("Error: Matrix dimensions must be positive\n");
        return 1;
    }
    if (n_runs <= 0) {
        printf("Error: Number of runs must be positive\n");
        return 1;
    }
    
    printf("=== Matrix Multiplication Benchmark ===\n");
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Initialization: %s\n", use_zero_init ? "zeros" : "random");
    printf("Number of runs: %d\n", n_runs);
    printf("Matrix sizes: A[%d×%d], B[%d×%d], C[%d×%d]\n", M, K, K, N, M, N);
    printf("Memory usage: A=%.2f MB, B=%.2f MB, C=%.2f MB, Total=%.2f MB\n",
           M * K * sizeof(float) / 1e6,
           K * N * sizeof(float) / 1e6,
           M * N * sizeof(float) / 1e6,
           (M * K + K * N + M * N) * sizeof(float) / 1e6);
    
    // Get GPU info
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("Error: No CUDA devices found\n");
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (Compute Capability %d.%d)\n", 
           prop.name, prop.major, prop.minor);
    printf("GPU Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("\n");
    
    // Seed random number generator
    srand(time(NULL));
    
    // Allocate host matrices
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));
    
    if (A == NULL || B == NULL || C == NULL) {
        printf("Error: Failed to allocate memory for matrices\n");
        free(A); free(B); free(C);
        return 1;
    }
    
    // Initialize matrices
    if (use_zero_init) {
        init_matrix_zero(A, M, K);
        init_matrix_zero(B, K, N);
    } else {
        init_matrix_random(A, M, K);
        init_matrix_random(B, K, N);
    }
    
    // Array to store timing results
    double *times = (double*)malloc(n_runs * sizeof(double));
    
    // ========================================================================
    // GPU BENCHMARKS
    // ========================================================================
    
    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy input matrices to GPU
    CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // ====================================================================
    // GPU Naive Kernel
    // ====================================================================
    printf("Running GPU Naive Kernel...\n");
    fflush(stdout);
    
    for (int run = 0; run < n_runs; run++) {
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
        
        Timer timer;
        timer_start(&timer);
        
        launch_matmul_gpu_naive(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        times[run] = timer_end(&timer);
        
        printf("  Run %d: %.6f seconds (%.2f GFLOPS)\n", 
               run + 1, times[run], (2.0 * M * N * K) / (times[run] * 1e9));
        fflush(stdout);
    }
    
    Stats stats = compute_stats(times, n_runs);
    print_stats("GPU Naive Kernel", stats, M, N, K);
    
    // Cleanup GPU memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    // Cleanup Host memory
    free(times);
    free(A);
    free(B);
    free(C);
    
    printf("Benchmark complete!\n");
    
    return 0;
}

