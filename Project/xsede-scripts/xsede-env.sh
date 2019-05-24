#!/bin/bash

cuda_version=9.2

module purge

module load gnutools
module load cuda/$cuda_version
module load python/2.7.10
module load pgi
module load mkl/11.1.2.144
module load mvapich2_gdr/2.1
module load openmpi_ib/1.8.4
module load gnu/4.9.2

case $cuda_version in
    8.0) export CUDA_ARCH_SM=sm_21 ;;
    9.2) export CUDA_ARCH_SM=sm_37 ;;
esac

export CUDA_HOME=/usr/local/cuda-$cuda_version
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
