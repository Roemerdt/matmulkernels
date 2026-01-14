# Matrix Multiplication Benchmark Results

Performance benchmarks for matrix multiplication implementations on NVIDIA Grace Hopper (GH200) superchip.

## System Specifications

- **CPU**: ARM Neoverse V2 (Grace)
- **GPU**: NVIDIA H200
- **Matrix Size**: 4096 × 4096
- **Precision**: FP32 (single precision)
- **Number of Runs**: 5

## Benchmark Results

### CPU (Grace) - Naive Implementation

| Metric | Time (seconds) | Performance (GFLOPS) |
|--------|----------------|----------------------|
| Min    | 172.780842     | 0.80                 |
| Max    | 173.093479     | 0.79                 |
| Avg    | 172.916551     | 0.79                 |
| Median | 172.834183     | 0.80                 |

### GPU (H200) - Naive Kernel

| Metric | Time (seconds) | Performance (GFLOPS) |
|--------|----------------|----------------------|
| Min    | 0.025348       | 5422.10              |
| Max    | 0.025422       | 5406.39              |
| Avg    | 0.025378       | 5415.72              |
| Median | 0.025357       | 5420.19              |

#### Performance Analysis
Achieved: ~5.4 TFLOPS while theoretical FP32 (non-tensor): ~60 TFLOPS, meaning only ~9% of peak theoretical performance
