#ifndef __COMMONDEF_H__
#define __COMMONDEF_H__

#define CUDA_RUNTIME_FN(expr) cuda_runtime_error_print_exit(expr, __LINE__, __FILE__, #expr)
#define CUDA_DRIVER_FN(expr) cuda_driver_error_print_exit(expr, __LINE__, __FILE__, #expr)
#define CUPTI_FN(expr) cupti_error_print_exit(expr, __LINE__, __FILE__, #expr)

typedef unsigned char bool_t;

enum {
	_False = 0,
	_True = 1
};

#ifdef __cplusplus
#define C_LINKAGE_START extern "C" {
#define C_LINKAGE_END }
#else
#define C_LINKAGE_START
#define C_LINKAGE_END
#endif // __cplusplus

#endif
