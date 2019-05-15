#ifndef __GPUMON_H__
#define __GPUMON_H__

#ifdef __cplusplus
#define EXTC extern "C"
#else
#define EXTC
#endif

#ifdef __CUDACC__
#define HOST __host__
#define DEV __device__
#else
#define HOST
#define DEV
#endif

typedef long long int clock64_t;

EXTC HOST void gpumon_host_start(int num_threads);

EXTC HOST void gpumon_host_end();
	
EXTC DEV void gpumon_device_start();

EXTC DEV void gpumon_device_end();

#endif //__GPUMON_H__
