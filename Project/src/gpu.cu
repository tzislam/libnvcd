#include "commondef.h"
#include "gpu.cuh"

#include <stdio.h>
#include <cuda.h>

C_LINKAGE_START	

__device__ bool assert_cond_impl(bool condition, const char* message, int line)
{
	if (!condition) {
		printf("DEVICE ASSERTION FAILURE: %s (line %i)\n", message, line);
	}

	return condition;
}
	
__device__ void print_thread_info()
{
	int thread =
		threadIdx.x +
		threadIdx.y * blockDim.x +
		threadIdx.z * blockDim.x * blockDim.y;
		
	printf("the thread: %i\n", thread);
}
	
__global__ void gpu_kernel()
{
	print_thread_info();
}

__host__ void gpu_test()
{
	dim3 grid(1, 1, 1);
	dim3 block(2, 2, 2);
	
	gpu_kernel<<<grid, block>>>();
}

__global__ void gpu_kernel_matrix_vec_int(int n,
										  int m,
										  int* q,
										  int* u,
										  int* v)
{
	printf("Kernel executed\n");
	
	int thread_row = threadIdx.x;

	if (GPU_ASSERT(thread_row < n)) {
		int c = 0;

		int k = 0;
			
		while (c < m) {
			k += q[thread_row * m + c] * u[c];
				
			c++;
		}

		v[thread_row] = k;
	}
}

__host__ void gpu_test_matrix_vec_mul()
{
	// TODO
}

C_LINKAGE_END
