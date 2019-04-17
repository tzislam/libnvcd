#include "common.h"
#include <nvml.h>

// main.cpp

int main()
{
	nvmlReturn_t ret = nvmlInit();

	printf("init ret: %i\n", ret);

	gpu_test();
	
	if (ret == NVML_SUCCESS) {
		ret = nvmlShutdown();

		printf("shutdown ret: %i\n", ret);
	}

	return 0;
}
