#ifndef __NVCI_H__
#define __NVCI_H__

#include "commondef.h"

// included here so the user may call NVCD_EXEC_KERNEL
// without needing to include any additional header files
#include "util.h"
#include <cuda.h>

C_LINKAGE_START

NVCD_EXPORT void nvcd_report();

NVCD_EXPORT void nvcd_init();

NVCD_EXPORT void nvcd_host_begin(int num_cuda_threads);

NVCD_EXPORT bool nvcd_host_finished();

NVCD_EXPORT void nvcd_host_end();

NVCD_EXPORT void nvcd_terminate();

#define NVCD_KERNEL_EXEC(kname, grid, block, ...)       \
  do {                                                  \
    while (!nvcd_host_finished()) {                     \
      kname<<<grid, block>>>(__VA_ARGS__);              \
      CUDA_RUNTIME_FN(cudaDeviceSynchronize());         \
    }                                                   \
  } while (0)

C_LINKAGE_END

#endif // __NVCI_H__
