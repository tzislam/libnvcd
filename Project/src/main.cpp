#include "common.h"

int main()
{
	gpu_test();
	
	CUDA_RUNTIME_FN(cudaDeviceSynchronize());

	return 0;
}
