#include "commondef.h"
#include "gpu.cuh"
#include "util.h"

#include <stdio.h>
#include <cuda.h>



/* 
 * template for taking the dot product of a vector with the row of a matrix 
 *
 * tmplRow: the row of the matrix
 * tmplM: the amount of columns the matrix has
 * tmplType: the scalar type of the matrix and the vectors
 * tmplQ: pointer of tmplType to the input matrix's buffer
 * tmplU: pointer of tmplType to the input vector's buffer
 * tmplV: pointer of tmplType to the output vector's buffer
 */ 
#define vec_mat_dot_tmpl(tmplRow, tmplM, tmplType, tmplQ, tmplU, tmplV)	\
	do {																\
		int c = 0;														\
		int __m = (tmplM);												\
		int __row = (tmplRow);											\
		tmplType k = 0;													\
		while (c < (tmplM)) {											\
			k += (tmplQ)[__row * __m + c] = (tmplU)[c];					\
			c++;														\
		}																\
		(tmplV)[__row] = k;												\
	} while (0)


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

__global__ void gpu_kernel_matrix_vec_int(int n,
										  int m,
										  int* q,
										  int* u,
										  int* v)
{	
	int thread_row = threadIdx.x;

	printf("Kernel executed: %i\n", thread_row);
	
	if (GPU_ASSERT(thread_row < n)) {
		vec_mat_dot_tmpl(thread_row, m, int, q, u, v);
	}
}

__host__ void gpu_test()
{
	dim3 grid(1, 1, 1);
	dim3 block(2, 2, 2);
	
	gpu_kernel<<<grid, block>>>();
}
 
static inline __host__ int* cuda_alloci(size_t num)
{
	void* memory = NULL;
	size_t memorysz = num * sizeof(int);

	CUDA_RUNTIME_FN(cudaMalloc(&memory, memorysz));
	CUDA_RUNTIME_FN(cudaMemset(memory, 0, memorysz));
	
	return (int*) memory;
}

static __host__ void cpu_matrix_vec_mul(int n, int m, int* q, int* u, int* v)
{
	for (int r = 0; r < n; ++r) {
		vec_mat_dot_tmpl(r, m, int, q, u, v);
	}
}

__host__ void gpu_test_matrix_vec_mul()
{
	dim3 grid(1, 1, 1);

	int n = 10; /* rows */
	int m = 10; /* columns */

	dim3 block(n, 1, 1);
	
	size_t msize = (size_t)(n * m);
	size_t vsize = (size_t)m;

	/* parallel q, u, v */
	int* q = cuda_alloci(msize); 
	int* u = cuda_alloci(vsize);
	int* v = cuda_alloci(vsize);

	/* serial q, u, v */
	int* sq = (int*) alloc_or_die(msize * sizeof(int));
	int* su = (int*) alloc_or_die(vsize * sizeof(int));
	int* sv = (int*) alloc_or_die(vsize * sizeof(int));

	/* client-side device result */
	int* hv = (int*) alloc_or_die(vsize * sizeof(int));
	
	int rmin = 5;
	int rmax = 50;
	
	for (size_t i = 0; i < msize; ++i) {
		sq[i] = random_nexti(rmin, rmax);
	}

	for (size_t i = 0; i < vsize; ++i) {
		su[i] = random_nexti(rmin, rmax);
	}

	CUDA_RUNTIME_FN(cudaMemcpy(q, sq, msize * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_RUNTIME_FN(cudaMemcpy(u, su, vsize * sizeof(int), cudaMemcpyHostToDevice));
	
	gpu_kernel_matrix_vec_int<<<grid, block>>>(n, m, q, u, v);

	CUDA_RUNTIME_FN(cudaDeviceSynchronize());

	CUDA_RUNTIME_FN(cudaMemcpy(hv, v, vsize * sizeof(int), cudaMemcpyDeviceToHost));

	cpu_matrix_vec_mul(n, m, sq, su, sv);

	bool equal = true;
	
	for (size_t i = 0; i < vsize && equal == true; ++i) {
		equal = hv[i] == sv[i];
	}

	if (equal == true) {
		puts("gpu_test_matrix_vec_mul: success");
	} else {
		puts("gpu_test_matrix_vec_mul: failure");
	}
}

C_LINKAGE_END
