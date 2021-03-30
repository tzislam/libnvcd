#!/bin/bash

# this is the installation path for the 11.2 version on Ubuntu 20.04, installed via the corresponding deb package
# which can be found at https://developer.nvidia.com/cuda-downloads
cuda_v=11.2.0
cuda_r=/usr/local/cuda-${cuda_v}

export CUPTI=${cuda_r}/extras/CUPTI
export CUDA_HOME=${cuda_r}

# We choose sm_50 as a default since it's almost guaranteed to be supported on the host's GPU
# in this case.
export CUDA_ARCH_SM=sm_50

# Avoid errors; if after sourcing this script there's issues related to preloaded binaries,
# comment this.
unset LD_PRELOAD
