#include "util.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

C_LINKAGE_START

void* alloc_or_die(size_t size)
{
	void* p = malloc(size);

	if (p == NULL) {
		perror("malloc failure");
		exit(1);
	}

	return p;
}

int random_nexti(int rmin, int rmax)
{
	srand(time(NULL));
	
	return rmin + rand()  % (rmax - rmin);
}

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
		const char* error_string = NULL;
		
		cuptiGetResultString(status, &error_string);
			
		printf("CUPTI: %s:%i:'%s' failed. [Reason] %s\n",
			   file,
			   line,
			   expr,
			   error_string);
			
		exit(status);
	}
}

void assert_impl(bool cond, const char* expr, const char* file, int line)
{
	if (!cond) {
		printf("ASSERT failure: \"%s\" @ %s:%i\n", expr, file, line);
		exit(1);
	}
}

C_LINKAGE_END
