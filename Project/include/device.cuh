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

#ifdef __CUDACC__
namespace nvcd {
  
  
}
#endif // __CUDACC__

EXTC NVCD_EXPORT DEV void nvcd_device_begin(int thread);
EXTC NVCD_EXPORT DEV void nvcd_device_end(int thread);

EXTC NVCD_EXPORT HOST void nvcd_device_init_mem(int num_threads);
EXTC NVCD_EXPORT HOST void nvcd_device_free_mem();

EXTC NVCD_EXPORT HOST void nvcd_device_get_ttime(clock64_t* out);
EXTC NVCD_EXPORT HOST void nvcd_device_get_smids(unsigned* out);

EXTC NVCD_EXPORT GLOBAL void nvcd_kernel_test();

EXTC NVCD_EXPORT HOST void nvcd_kernel_test_call(int num_threads);

#endif // __DEVICE_CUH__
