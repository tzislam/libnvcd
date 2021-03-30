#!/bin/bash

#just a placeholder for now;
#since running on other
#versions will likely be desired
cuda_v=11.2.0
cuda_r=/usr/local/cuda-${cuda_v}
repo=$(dirname $(realpath $BASH_SOURCE))/../..

export LD_LIBRARY_PATH=${cuda_r}/lib64:${repo}/bin:$LD_LIBRARY_PATH
export LD_PRELOAD=${repo}/bin/libnvcdhook.so:$LD_PRELOAD

