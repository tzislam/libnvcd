#include "common.h"
//#include <nvml.h>

// main.cpp

int main()
{
	//	nvmlReturn_t ret = nvmlInit();

	//printf("init ret: %i\n", ret);

	gpu_test();
	CUDA_FN(cudaDeviceSynchronize());

#if 0
	if (ret == NVML_SUCCESS) {
		//	ret = nvmlShutdown();

		printf("shutdown ret: %i\n", ret);
	}
	#endif

	return 0;
}
