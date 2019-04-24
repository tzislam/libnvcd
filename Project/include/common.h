#ifndef __COMMON_H__
#define __COMMON_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <stdio.h>
#include <stdlib.h>

#define GPU_FN __device__
#define GPU_INL_FN static inline GPU_FN
#define GPU_KERN_FN __global__

#define GPU_CLIENT_FN __host__

#define GPU_KERN_DECL extern "C" GPU_KERN_FN
#define GPU_CLIENT_DECL extern "C" GPU_CLIENT_FN

// these are implemented in src/gpu_error.cpp
extern "C" {

	void cuda_runtime_error_print_exit(cudaError_t status,
									   int line,
									   const char* file,
									   const char* expr);

	void cuda_driver_error_print_exit(CUresult status,
									  int line,
									  const char* file,
									  const char* expr);
	
	void cupti_error_print_exit(CUptiResult status,
								int line,
								const char* file,
								const char* expr);
}

#define CUDA_RUNTIME_FN(expr) cuda_runtime_error_print_exit(expr, __LINE__, __FILE__, #expr)

#define CUDA_DRIVER_FN(expr) cuda_driver_error_print_exut(expr, __LINE__, __FILE__, #expr)

#define CUPTI_FN(expr) cupti_error_print_exit(expr, __LINE__, __FILE__, #expr)

GPU_KERN_DECL void gpu_kernel();
GPU_CLIENT_DECL void gpu_test();

#endif
