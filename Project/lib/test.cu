#include "gpumon.h"
#include "gpumon.cuh"
#include "util.h"
#include <stdio.h>

#ifdef SANDBOX
__device__ int* const_symbol;

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/***************/
/* TEST KERNEL */
/***************/
__global__ void kernel() {

	printf("Address of symbol from device = %p\n", &const_symbol);
	const_symbol[0] = 2;
	const_symbol[1] = 3;
	const_symbol[2] = 3;
}
#endif

int main() {
#ifdef SANDBOX
	const int N = 16;

	void* d_const_sym = __cuda_zalloc_sym(sizeof(int) * 3, const_symbol);
	
	kernel<<<1,1>>>();

	CUDA_RUNTIME_FN(cudaDeviceSynchronize());
	
	int k[3];

	CUDA_RUNTIME_FN(cudaMemcpy(k, d_const_sym, sizeof(int) * 3, cudaMemcpyDeviceToHost));

	for(int i = 0; i < 3; ++i) {
		printf("k[%i] = %i\n", i, k[i]);
	}
#else
	const int N = 1024 << 1;
	gpumon_host_start(N);

	gpumon_kernel_test_call(N);
	
	gpumon_host_end();
#endif
	return 0;
}
