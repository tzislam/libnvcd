#ifndef __GPUMON_CUH__
#define __GPUMON_CUH__

#include "gpumon.h"

EXTC DEV void gpumon_device_start(int thread);
EXTC DEV void gpumon_device_end(int thread);

EXTC HOST void gpumon_init_device_mem(int num_threads);
EXTC HOST void gpumon_free_device_mem();

EXTC HOST void gpumon_get_device_ttime(clock64_t* out);

EXTC GLOBAL void gpumon_kernel_test();

EXTC HOST void gpumon_kernel_test_call(int num_threads);

#endif
