#!/bin/bash

#just a placeholder for now;
#since running on other
#versions will likely be desired
cuda_v=10.2.89
cuda_r=/usr/tce/packages/cuda/cuda-${cuda_v}

# NOTE:
# usually CUPTI is located in ${CUDA_HOME}/extras/CUPTI;
# however on Lassen they store all of the libraries in ${CUDA_HOME}/lib64.
# The makefile uses these common values as the default, so we override them here
# to accomodate
export CUPTI=${cuda_r}
export CUDA_HOME=${cuda_r}

module load gcc/8.3.1
module load cuda/${cuda_v}

unset LD_PRELOAD
