#ifndef __NVCD_H__
#define __NVCD_H__

#include "commondef.h"
#include <cuda.h>

C_LINKAGE_START

typedef struct cupti_event_data cupti_event_data_t;

NVCD_EXPORT void nvcd_init_events(CUdevice device, CUcontext context);

NVCD_EXPORT void nvcd_calc_metrics();

NVCD_EXPORT void nvcd_free_events();

NVCD_EXPORT cupti_event_data_t* nvcd_get_events();

C_LINKAGE_END

#endif // __NVCD_H__
