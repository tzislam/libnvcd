#include "gpumon.h"

#include <cuda.h>

static __device__ clock64_t* dev_tstart = nullptr;
static __device__ clock64_t* dev_ttime = nullptr;



