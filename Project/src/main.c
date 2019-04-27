#include "commondef.h"
#include "gpu.h"

int main()
{
	gpu_test_matrix_vec_mul();
	
	gpu_test();
	
	CUDA_RUNTIME_FN(cudaDeviceSynchronize());

	return 0;
}
