#include "util.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void* zalloc(size_t sz)
{
	void* p = malloc(sz);

	if (p != NULL) {
		memset(p, 0, sz);
	} else {
		puts("OOM");
	}

	/* set here for testing purposes; 
	   should not be relied upon for any
	   real production build */

	return p;
}

void* assert_not_null_impl(void* p, const char* expr, const char* file, int line)
{
	if (p == NULL) {
		assert_impl(false, expr, file, line);
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

#ifdef __cplusplus
}
#endif
