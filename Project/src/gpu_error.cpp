#include "common.h"

extern "C" {
	void cuda_runtime_error_print_exit(cudaError_t status,
									   int line,
									   const char* file,
									   const char* expr)
	{
		if (status != cudaSuccess) {
			printf("CUDA RUNTIME: %s:%i:'%s' failed. [Reason] %s:%s\n",
				   file,
				   line,
				   expr,
				   cudaGetErrorName(status),
				   cudaGetErrorString(status));
			
			exit(status);
		}
	}

	void cuda_driver_error_print_exit(CUresult status,
									  int line,
									  const char* file,
									  const char* expr)
	{
		if (status != CUDA_SUCCESS) {
			printf("CUDA DRIVER: %s:%i:'%s' failed. [Reason] %i\n",
				   file,
				   line,
				   expr,
				   status);
			
			exit(status);
		}
	}
	
	void cupti_error_print_exit(CUptiResult status,
								int line,
								const char* file,
								const char* expr)
	{
		if (status != CUPTI_SUCCESS) {
			const char* error_string = nullptr;
			cuptiGetResultString(status, &error_string);
			
			printf("CUPTI: %s:%i:'%s' failed. [Reason] %s\n",
				   file,
				   line,
				   expr,
				   error_string);
			
			exit(status);
		}
	}
}
