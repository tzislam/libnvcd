#ifndef __GPU_CUH__
#define __GPU_CUH__

#include "commondef.h"

#include <stdio.h>

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
	printf("Kernel executed\n");
	
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


#endif
