#include <nvml.h>
#include <stdio.h>

int main() {
	nvmlReturn_t ret = nvmlInit();

	printf("init ret: %i\n", ret);
	
	if (ret == NVML_SUCCESS) {
		ret = nvmlShutdown();

		printf("shutdown ret: %i\n", ret);
	}

	return 0;
}
