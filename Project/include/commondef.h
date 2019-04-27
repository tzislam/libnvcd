#ifndef __COMMONDEF_H__
#define __COMMONDEF_H__

#define CUDA_RUNTIME_FN(expr) cuda_runtime_error_print_exit(expr, __LINE__, __FILE__, #expr)
#define CUDA_DRIVER_FN(expr) cuda_driver_error_print_exit(expr, __LINE__, __FILE__, #expr)
#define CUPTI_FN(expr) cupti_error_print_exit(expr, __LINE__, __FILE__, #expr)

#define GPU_FN __device__
#define GPU_INL_FN static inline GPU_FN
#define GPU_KERN_FN __global__

#define GPU_CLIENT_FN __host__

#define GPU_API extern "C"

#if __cplusplus < 201103L
#pragma error "This should not be happening"
#define nullptr NULL
#endif

#endif
