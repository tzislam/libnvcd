#ifndef __COMMONDEF_H__
#define __COMMONDEF_H__

#define CUDA_RUNTIME_FN(expr) cuda_runtime_error_print_exit(expr, __LINE__, __FILE__, #expr)
#define CUDA_DRIVER_FN(expr) cuda_driver_error_print_exit(expr, __LINE__, __FILE__, #expr)
#define CUPTI_FN(expr) cupti_error_print_exit(expr, __LINE__, __FILE__, #expr)

/*
 * NOTE: bool appears to automatically be defined for CUDA;
 * because nvcc proxies through gcc, the C source modules
 * need to have stdbool.h included.
 */

#include <stdint.h>

#if !defined(__CUDACC__) && !defined(__cplusplus)
#include <stdbool.h>
#endif

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

#define ASSERT(cond) assert_impl((cond), #cond, __FILE__, __LINE__)

#define NOT_NULL(p_expr) assert_not_null_impl((p_expr), #p_expr, __FILE__, __LINE__) 

#define EVENT_ID_UNSET (-1)

#ifdef __cplusplus
#define C_LINKAGE_START extern "C" {
#define C_LINKAGE_END }
#else
#define C_LINKAGE_START
#define C_LINKAGE_END
#endif // __cplusplus

#endif
