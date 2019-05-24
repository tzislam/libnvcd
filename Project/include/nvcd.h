#ifndef __NVCI_H__
#define __NVCI_H__

#include "commondef.h"

C_LINKAGE_START

typedef bool (*nvcd_host_exec_fn_t)(void* userdata);

NVCD_EXPORT void nvcd_init();

NVCD_EXPORT void nvcd_host_begin(nvcd_host_exec_fn_t callback, void* userdata);
NVCD_EXPORT void nvcd_host_end();

NVCD_EXPORT void nvcd_terminate();

C_LINKAGE_END

#endif // __NVCI_H__
