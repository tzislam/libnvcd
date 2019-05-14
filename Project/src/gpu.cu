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
#define vec_mat_dot_tmpl(tmplRow, tmplM, tmplType, tmplQ, tmplU, tmplV) \
	do {																																	\
		int c = 0;																													\
		int __m = (tmplM);																									\
		int __row = (tmplRow);																							\
		int* __Q = (tmplQ);																									\
		int* __U = (tmplU);																									\
		int* __V = (tmplV);																									\
		tmplType k = 0;																											\
		while (c < __m) {																										\
			k += __Q[__row * __m + c] * __U[c];																\
			c++;																															\
		}																																		\
		__V[__row] = k;																											\
	} while (0)

#define cuda_alloc_tmpl(tmplType, in_tmplN, out_tmplMemory)	\
	do {																											\
		void* memory = NULL;																		\
		size_t memorysz = (in_tmplN) * sizeof(tmplType);				\
																														\
		CUDA_RUNTIME_FN(cudaMalloc(&memory, memorysz));					\
		ASSERT(memory != NULL);																	\
		CUDA_RUNTIME_FN(cudaMemset(memory, 0, memorysz));				\
																														\
		(out_tmplMemory) = (tmplType*) memory;									\
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
																					int* v,
																					clock64_t* d_times)
{	
	int thread_row = threadIdx.x;

	//	printf("Kernel executed: %i\n", thread_row);
	
	if (GPU_ASSERT(thread_row < n)) {
		clock64_t start = clock64();
		
		vec_mat_dot_tmpl(thread_row, m, int, q, u, v);

		clock64_t time = clock64() - start;

		d_times[thread_row] = time;
	}
}

__host__ void gpu_test()
{
	dim3 grid(1, 1, 1);
	dim3 block(2, 2, 2);
	
	gpu_kernel<<<grid, block>>>();
}

static inline __host__ int* cuda_alloci(size_t num) {
	int* imem = NULL;
	cuda_alloc_tmpl(int, num, imem);
	return imem;
}

static inline __host__ int64_t* cuda_alloci64(size_t num) {
	int64_t* mem = NULL;
	cuda_alloc_tmpl(int64_t, num, mem);
	return mem;
}

static __host__ void cpu_matrix_vec_mul(int n, int m, int* q, int* u, int* v)
{
	for (int r = 0; r < n; r++) {
		int c = 0;																																																						

		int k = 0;																											
		while (c < m) {																										
			k += q[r * m + c] * u[c];																
			c++;																															
		}
		
		v[r] = k;
	}
}

__host__ void gpu_test_matrix_vec_mul(int num_threads, clock64_t* h_exec_times)
{
	dim3 grid(1, 1, 1);

	int n = num_threads; /* rows */
	int m = 10; /* columns */

	dim3 block(n, 1, 1);
	
	size_t qsize = (size_t)(n * m); /* matrix size */
	size_t usize = (size_t)m;
	size_t vsize = (size_t)n; /* vector size */

	/* parallel q, u, v */
	int* q = cuda_alloci(qsize); 
	int* u = cuda_alloci(usize);
	int* v = cuda_alloci(vsize);

	/* serial q, u, v */
	int* sq = (int*) NOT_NULL(zalloc(qsize * sizeof(int)));
	int* su = (int*) NOT_NULL(zalloc(usize * sizeof(int)));
	int* sv = (int*) NOT_NULL(zalloc(vsize * sizeof(int)));

	/* client-side device result */
	int* hv = (int*) NOT_NULL(zalloc(vsize * sizeof(int)));
	
	int rmin = 5;
	int rmax = 50;
	
	for (size_t i = 0; i < qsize; ++i) {
		sq[i] = random_nexti(rmin, rmax);
	}

	for (size_t i = 0; i < usize; ++i) {
		su[i] = random_nexti(rmin, rmax);
	}

	CUDA_RUNTIME_FN(cudaMemcpy(q, sq, qsize * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_RUNTIME_FN(cudaMemcpy(u, su, usize * sizeof(int), cudaMemcpyHostToDevice));

	{
		ASSERT(sizeof(clock64_t) == sizeof(int64_t));
		
		clock64_t* d_exec_times = (clock64_t*) cuda_alloci64((size_t) num_threads);

		gpu_kernel_matrix_vec_int<<<grid, block>>>(n, m, q, u, v, d_exec_times);

		CUDA_RUNTIME_FN(cudaMemcpy(h_exec_times,
															 d_exec_times,
															 sizeof(clock64_t) * num_threads,
															 cudaMemcpyDeviceToHost));

		CUDA_RUNTIME_FN(cudaFree(d_exec_times));
	}
	
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
	
	CUDA_RUNTIME_FN(cudaFree(q));
	CUDA_RUNTIME_FN(cudaFree(u));
	CUDA_RUNTIME_FN(cudaFree(v));
}

C_LINKAGE_END
