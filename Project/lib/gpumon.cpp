#include "gpumon.h"
#include "gpumon.cuh"
#include "util.h"

#include <vector>
#include <memory>
#include <stdio.h>

static std::vector<clock64_t> host_ttime;

EXTC HOST void gpumon_host_start(int num_threads) {
	host_ttime.clear();
	host_ttime.resize(num_threads, 0);

	CUDA_RUNTIME_FN(cudaSetDevice(0));
	
	gpumon_init_device_mem(num_threads);
}

EXTC HOST void gpumon_host_end() {
	CUDA_RUNTIME_FN(cudaDeviceSynchronize());
	
	gpumon_get_device_ttime(&host_ttime[0]);

	for (size_t i = 0; i < host_ttime.size(); ++i) {
		printf("[%lu] time = %lli \n", i, host_ttime.at(i));
	}
}
