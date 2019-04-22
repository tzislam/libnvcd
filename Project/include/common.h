#ifndef __COMMON_H__
#define __COMMON_H__

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#define GPU_FN __device__
#define GPU_INL_FN static inline GPU_FN
#define GPU_KERN_FN __global__

#define GPU_CLIENT_FN __host__

#define GPU_KERN_DECL extern "C" GPU_KERN_FN
#define GPU_CLIENT_DECL extern "C" GPU_CLIENT_FN


#if 0
extern "C" {

static inline void cuda_error_print_exit(cudaError_t status, int line, const char* expr)
{
	if (status != cudaSuccess) {
		printf("CUDA: %i:'%s' failed. [Reason] %s:%s\n", line, expr, cudaGetErrorName(status), cudaGetErrorString(status));
		exit(status);
	}
}
	
}
#endif

#define CUDA_FN(expr) //cuda_error_print_exit(expr, __LINE__, #expr)

GPU_KERN_DECL void gpu_kernel();
GPU_CLIENT_DECL void gpu_test();

#endif
