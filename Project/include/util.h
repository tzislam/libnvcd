#ifndef __UTIL_H__
#define __UTIL_H__

#include "nvcd/commondef.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <time.h>
#include <sys/time.h>

C_LINKAGE_START

void exit_msg(FILE* out, int error, const char* message, ...);

// Reallocates a buffer of size
// elem_size * (*current_length)
// to
// elem_size * (*current_length) * 2
// and zeros out the newly added memory.
//
// Returns the pointer returned by
// realloc (which may be NULL).
//
// Performs sanity checks on input.
void* double_buffer_size(void* buffer,
                         size_t elem_size,
                         size_t* current_length);

NVCD_EXPORT void* zalloc(size_t sz);

NVCD_EXPORT void safe_free(void** p); // safer, but not "safe"

NVCD_EXPORT int random_nexti(int rmin, int rmax);

NVCD_EXPORT void cuda_runtime_error_print_exit(cudaError_t status,
                                               int line,
                                               const char* file,
                                               const char* expr);

NVCD_EXPORT void cuda_driver_error_print_exit(CUresult status,
                                              int line,
                                              const char* file,
                                              const char* expr);
  
NVCD_EXPORT void cupti_error_print_exit(CUptiResult status,
                                        int line,
                                        const char* file,
                                        const char* expr);

NVCD_EXPORT void assert_impl(bool cond,
                             const char* expr,
                             const char* file,
                             int line);

NVCD_EXPORT void* assert_not_null_impl(void* p, const char* expr, const char* file, int line);


C_LINKAGE_END

#endif // __UTIL_H__
