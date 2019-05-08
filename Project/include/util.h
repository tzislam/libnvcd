#ifndef __UTIL_H__
#define __UTIL_H__

#include "commondef.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

C_LINKAGE_START

void* alloc_or_die(size_t size);

int random_nexti(int rmin, int rmax);

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

void assert_impl(bool cond,
								 const char* expr,
								 const char* file,
								 int line);

C_LINKAGE_END

#endif // __UTIL_H__
