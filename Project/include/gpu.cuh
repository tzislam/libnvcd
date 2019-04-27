#ifndef __GPU_H__
#define __GPU_H__

#include "common.h"

#include <stdio.h>

#ifdef __CUDA_ARCH__

#pragma message "CUDA CODE"

#define GPU_ASSERT(condition_expr) assert_cond_impl(condition_expr, #condition_expr, __LINE__) 

GPU_API GPU_FN bool assert_cond_impl(bool condition, const char* message, int line);

template <typename intType>
GPU_FN intType thread_index1() {
	return intType(threadIdx.x);
} 


/*
 * transforms an M x 1 vector u 
 * by an N x M matrix q
 * resulting in an N x 1 vector v
 *
 * qu = v 
 */
template <typename scalarType>
GPU_KERN_FN void gpu_kernel_matrix_vec_mul(int n,
										   int m,
										   scalarType* q,
										   scalarType* u,
										   scalarType* v)
{
	int thread_row = thread_index1<int>();

	if (GPU_ASSERT(thread_row < n)) {
		int c = 0;

		scalarType k = 0;
			
		while (c < m) {
			k += q[thread_row * m + c] * u[c];
				
			c++;
		}

		v[thread_row] = k;
	}
}

#else

#pragma message "NON CUDA CODE"

template <typename scalarType>
static inline GPU_KERN_FN void gpu_kernel_matrix_vec_mul(int n,
										   int m,
										   scalarType* q,
										   scalarType* u,
										   scalarType* v) {}

#include "common.inl"

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


#endif
