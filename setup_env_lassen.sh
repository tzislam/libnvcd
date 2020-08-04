#!/bin/bash

#just a placeholder for now;
#since running on other
#versions will likely be desired
cuda_v=10.1.243
cuda_r=/usr/tce/packages/cuda/cuda-${cuda_v}
repo=$(dirname $(realpath $BASH_SOURCE))

export LD_LIBRARY_PATH=${cuda_r}/extras/CUPTI/lib64:${cuda_r}/lib64:${repo}/bin

