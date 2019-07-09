#!/bin/bash

cuda_version=9.2

export NVCD_HOME=$HOME/cs415/Project
export NVCD_BIN_PATH=$NVCD_HOME/bin
export NVCD_INCLUDE_PATH=$NVCD_HOME/include

export LD_LIBRARY_PATH=$NVCD_BIN_PATH:$LD_LIBRARY_PATH

module purge

module load gnutools
module load cuda/$cuda_version
module load python/2.7.10
module load pgi
module load mkl/11.1.2.144
module load mvapich2_gdr/2.1
module load openmpi_ib/1.8.4
module load gnu/4.9.2

arch=

case $cuda_version in
    8.0) arch=21 ;;
    
    9.2) arch=37 ;;
esac

export CUDA_ARCH_SM=sm_${arch}
export CUDA_ARCH_COMPUTE=compute_${arch}

export CUDA_HOME=/usr/local/cuda-$cuda_version

export CUPTI_INCLUDE=$CUDA_HOME/extras/CUPTI/include
export CUPTI_LIB=$CUDA_HOME/extras/CUPTI/lib64

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUPTI_LIB:$LD_LIBRARY_PATH

export JOB_OUTPUT_DIR=$NVCD_HOME/xsede-scripts/job-output


mkdir -p $JOB_OUTPUT_DIR
