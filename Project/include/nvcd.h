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

NVCD_EXPORT void nvcd_host_begin();

NVCD_EXPORT bool nvcd_host_finished();

NVCD_EXPORT void nvcd_host_end();

NVCD_EXPORT void nvcd_terminate();

#define NVCD_EXEC_KERNEL(kernel_invoke_expr)    \
  do {                                          \
    nvcd_host_begin();                          \
    while (!nvcd_host_finished()) {             \
      (kernel_invoke_expr);                     \
      CUDA_RUNTIME_FN(cudaDeviceSynchronize()); \
    }                                           \
    nvcd_host_end();                            \
  } while (0)

C_LINKAGE_END

#endif // __NVCI_H__
