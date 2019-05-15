#include "gpumon.h"
#include "gpumon.cuh"

int main() {
	gpumon_host_start(1024);

	gpumon_kernel_test_call(1024);
	
	gpumon_host_end();

	return 0;
}
