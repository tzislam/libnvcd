#include "common.h"

extern "C" {
	GPU_FN void assert_cond_impl(bool condition, const char* message)
	{
		if (!condition) {
			printf("DEVICE ERROR: %s\n", message);
		}
	}
	
	GPU_FN void print_thread_info()
	{
		int thread = threadIdx.x + threadIdx.y * blockDim.x;
		
		printf("the thread: %i\n", thread);
	}
	
	GPU_KERN_FN void gpu_kernel()
	{
		print_thread_info();
	}

	GPU_CLIENT_FN void gpu_test()
	{
		dim3 grid(1, 1, 1);
		dim3 block(2, 2, 1);
	
		gpu_kernel<<<grid, block>>>();
	}
}
