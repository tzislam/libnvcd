#include "common.h"

#include "gpu.cuh"

extern "C" {	

	GPU_FN bool assert_cond_impl(bool condition, const char* message, int line)
	{
		if (!condition) {
			printf("DEVICE ASSERTION FAILURE: %s (line %i)\n", message, line);
		}

		return condition;
	}
	
	GPU_FN void print_thread_info()
	{
		int thread =
			threadIdx.x +
			threadIdx.y * blockDim.x +
			threadIdx.z * blockDim.x * blockDim.y;
		
		printf("the thread: %i\n", thread);
	}
	
	GPU_KERN_FN void gpu_kernel()
	{
		print_thread_info();
	}

	GPU_CLIENT_FN void gpu_test()
	{
		dim3 grid(1, 1, 1);
		dim3 block(2, 2, 2);
	
		gpu_kernel<<<grid, block>>>();
	}
}
