#include "gpumon.h"

#include "util.h"

static __device__ clock64_t* dev_tstart = nullptr;
static __device__ clock64_t* dev_ttime = nullptr;

EXTC HOST void gpumon_init_device_mem(int num_threads) {
	
}


