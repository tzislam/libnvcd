#ifndef __DEVICE_CUH__
#define __DEVICE_CUH__

#include "commondef.h"

#ifdef __cplusplus
#define EXTC extern "C"
#else
#define EXTC
#endif

#ifdef __CUDACC__
#define DEV __device__
#define HOST __host__
#define GLOBAL __global__
#else
#define DEV
#define HOST
#define GLOBAL
#endif

EXTC DEV void nvcd_device_start(int thread);
EXTC DEV void nvcd_device_end(int thread);

EXTC HOST void nvcd_init_device_mem(int num_threads);
EXTC HOST void nvcd_free_device_mem();

EXTC HOST void nvcd_get_device_ttime(clock64_t* out);
EXTC HOST void nvcd_get_device_smids(unsigned* out);

EXTC GLOBAL void nvcd_kernel_test();

EXTC HOST void nvcd_kernel_test_call(int num_threads);

#endif // __DEVICE_CUH__
