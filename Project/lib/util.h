#ifndef __UTIL_H__
#define __UTIL_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#define CUDA_RUNTIME_FN(expr) cuda_runtime_error_print_exit(expr, __LINE__, __FILE__, #expr)
#define CUDA_DRIVER_FN(expr) cuda_driver_error_print_exit(expr, __LINE__, __FILE__, #expr)
#define CUPTI_FN(expr) cupti_error_print_exit(expr, __LINE__, __FILE__, #expr)

#ifdef __cplusplus
extern "C" {
#endif

void* zalloc(size_t sz);

int random_nexti(int rmin, int rmax);

void cuda_runtime_error_print_exit(cudaError_t status,
                                   int line,
                                   const char* file,
                                   const char* expr);

void cuda_driver_error_print_exit(CUresult status,
                                  int line,
                                  const char* file,
                                  const char* expr);
  
void cupti_error_print_exit(CUptiResult status,
                            int line,
                            const char* file,
                            const char* expr);

void assert_impl(bool cond,
                 const char* expr,
                 const char* file,
                 int line);

void* assert_not_null_impl(void* p, const char* expr, const char* file, int line);

#ifdef __cplusplus
}
#endif
  
#endif // __UTIL_H__
