#ifndef __NVCD_H__
#define __NVCD_H__

#include "commondef.h"
#include <cuda.h>

C_LINKAGE_START

typedef CUuuid uuid_t;

typedef struct nvcd {
  CUdevice* devices;
  CUcontext* contexts;
  bool32_t* contexts_ext;
  char** device_names;
  uuid_t* device_uuids;
  
  int num_devices;
  
  bool32_t initialized;
  bool32_t opt_verbose_output; 
  bool32_t opt_diagnostic_output;
} nvcd_t;

typedef struct cupti_event_data cupti_event_data_t;

// see nvcd.cuh
NVCD_EXPORT extern nvcd_t g_nvcd;

NVCD_EXPORT void nvcd_init_cuda();

NVCD_EXPORT void nvcd_init_events(CUdevice device, CUcontext context);

NVCD_EXPORT void nvcd_calc_metrics();


NVCD_EXPORT void nvcd_reset_event_data();

NVCD_EXPORT cupti_event_data_t* nvcd_get_events();

C_LINKAGE_END

#endif // __NVCD_H__
