#ifndef __COMMON_H__
#define __COMMON_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include "commondef.h"

// these are implemented in src/util.cpp
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

GPU_API GPU_KERN_FN void gpu_kernel();
GPU_API GPU_CLIENT_FN void gpu_test();

#endif
