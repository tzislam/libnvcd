#ifndef __GPU_H__
#define __GPU_H__

#include "commondef.h"
#include "util.h"
#include <stdio.h>

GPU_API GPU_KERN_FN void gpu_kernel();

GPU_API GPU_CLIENT_FN void gpu_test();

template <typename scalarType>
static inline GPU_KERN_FN void gpu_kernel_matrix_vec_mul(int n,
										   int m,
										   scalarType* q,
										   scalarType* u,
										   scalarType* v) {}

template <typename scalarType, int N, int M, int min, int max>
static inline GPU_CLIENT_FN void gpu_test_matrix_vec_mul()
{
	scalarType* matrix = new scalarType[N * M];
	scalarType* in_vector = new scalarType[M];
		
	scalarType* out_vector = new scalarType[N]();
	
	std::string x = util::to_string(out_vector, N);

	printf("out_vector: %s\n", x.c_str());
}

#endif
