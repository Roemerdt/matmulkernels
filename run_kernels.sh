#!/bin/bash
#SBATCH --job-name=kernels
#SBATCH --constraint=nvidia_grace
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=output/cuda_%j.out
#SBATCH --error=output/cuda_%j.err

# Setup CUDA environment
export PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0/lib64:$LD_LIBRARY_PATH

nvcc -O3 -arch=sm_90 -o kernels kernels.cu

./kernels "$@"
