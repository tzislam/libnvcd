#ifndef __DEVICE_CUH__
#define __DEVICE_CUH__

#include "commondef.h"

#ifdef __cplusplus
#define EXTC extern "C"
#else
#define EXTC
#endif

#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime.h>

#define DEV __device__
#define HOST __host__
#define GLOBAL __global__

#define NVCD_DEV_EXPORT EXTC NVCD_EXPORT DEV

#else
#define DEV
#define HOST
#define GLOBAL
#endif

#ifdef __CUDACC__
namespace detail {
  DEV clock64_t* dev_tstart;
  DEV clock64_t* dev_ttime;
  DEV int* dev_num_iter;
  DEV uint* dev_smids;
}

// see https://devtalk.nvidia.com/default/topic/481465/any-way-to-know-on-which-sm-a-thread-is-running-/

NVCD_DEV_EXPORT uint get_smid() {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

NVCD_DEV_EXPORT void nvcd_device_begin(int thread) {
  detail::dev_tstart[thread] = clock64(); 
}

NVCD_DEV_EXPORT void nvcd_device_end(int thread) {
  detail::dev_ttime[thread] = clock64() - detail::dev_tstart[thread];
  detail::dev_smids[thread] = get_smid();
}

#endif // __CUDACC__

EXTC NVCD_EXPORT HOST void nvcd_device_init_mem(int num_threads);
EXTC NVCD_EXPORT HOST void nvcd_device_free_mem();

EXTC NVCD_EXPORT HOST void nvcd_device_get_ttime(clock64_t* out);
EXTC NVCD_EXPORT HOST void nvcd_device_get_smids(unsigned* out);

EXTC NVCD_EXPORT GLOBAL void nvcd_kernel_test();

EXTC NVCD_EXPORT HOST void nvcd_kernel_test_call(int num_threads);

#endif // __DEVICE_CUH__
