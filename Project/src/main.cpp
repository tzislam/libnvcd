#include "common.h"
#include "gpu.h"

int main()
{
	gpu_test_matrix_vec_mul<int, 2, 2, 0, 1000>();
	
	gpu_test();
	
	CUDA_RUNTIME_FN(cudaDeviceSynchronize());

	return 0;
}
