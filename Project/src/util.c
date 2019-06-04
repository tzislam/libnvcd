#include "nvcd/util.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdarg.h>

C_LINKAGE_START

void exit_msg(FILE* out, int error, const char* message, ...) {
  char* buffer = calloc(strlen(message) + 128, sizeof(char));
  
  va_list ap;
  va_start(ap, message);
  vsprintf(buffer, message, ap);
  va_end(ap);

  fprintf(out, "EXIT TRIGGERED. Reason: \"%s\". Code: 0x%x\n", buffer, error);
  
  exit(error);
}

#define SANITY_CHECK_TOTAL_SIZE 0x07FFFFFF
#define SANITY_CHECK_ELEM (16 << 6)

void* double_buffer_size(void* buffer,
                         size_t elem_size,
                         size_t* current_length) {
  ASSERT(buffer != NULL);
  ASSERT(current_length != NULL);
  ASSERT(elem_size > 0 && elem_size < SANITY_CHECK_ELEM);
  ASSERT((elem_size & 1) == 0);
  ASSERT(*current_length > 0);
  
  size_t new_length = *current_length << 1;
  size_t new_size = new_length * elem_size;

  ASSERT(new_size < SANITY_CHECK_TOTAL_SIZE);

  void* newp = realloc(buffer, new_size);

  if (newp != NULL) {   
    uint8_t* bptr = (uint8_t*)newp;

    size_t half_sz = elem_size * (*current_length);

    if (half_sz < new_length) {
      ASSERT(half_sz == new_length >> 1);
      // zero out uninitialized memory
      memset((void*)(&bptr[half_sz]), 0, half_sz);
    }
    
    *current_length = new_length;
  } else {
    printf("WARNING: realloc failure for\n"
           "\tcurrent_length = 0x%" PRIx64 "\n"
           "\telem_size = 0x%" PRIx64 "\n"
           "\tbuffer = %p\n",
           *current_length,
           elem_size,
           buffer);
  }

  return newp;
}

NVCD_EXPORT void* zalloc(size_t sz) {
  void* p = malloc(sz);

  if (p != NULL) {
    memset(p, 0, sz);
  } else {
    printf("WARNING: OOM in zalloc for size: 0x%" PRIx64 "\n", sz);
  }

  /* set here for testing purposes; 
     should not be relied upon for any
     real production build */

  return p;
}

NVCD_EXPORT void safe_free(void** p) {
  if (*p != NULL) {
    free(*p);
    *p = NULL;
  }
}

NVCD_EXPORT void free_strlist(char** list, size_t length) {
  for (size_t i = 0; i < length; ++i) {
    safe_free((void**) &list[i]);
  }

  safe_free((void**) &list);
}

NVCD_EXPORT void* assert_not_null_impl(void* p, const char* expr, const char* file, int line) {
  if (p == NULL) {
    assert_impl(false, expr, file, line);
  }
  
  return p;
}

NVCD_EXPORT int random_nexti(int rmin, int rmax) {
  srand(time(NULL));
  
  return rmin + rand()  % (rmax - rmin);
}

NVCD_EXPORT void cuda_runtime_error_print_exit(cudaError_t status,
                                               int line,
                                               const char* file,
                                               const char* expr) {
  if (status != cudaSuccess) {
    printf("CUDA RUNTIME: %s:%i:'%s' failed. [Reason] %s:%s\n",
           file,
           line,
           expr,
           cudaGetErrorName(status),
           cudaGetErrorString(status));
      
    exit(ECUDA_RUNTIME);
  }
}

NVCD_EXPORT void cuda_driver_error_print_exit(CUresult status,
                                              int line,
                                              const char* file,
                                              const char* expr) {
  if (status != CUDA_SUCCESS) {
    printf("CUDA DRIVER: %s:%i:'%s' failed. [Reason] %i\n",
           file,
           line,
           expr,
           status);
      
    exit(ECUDA_DRIVER);
  }
}
  
NVCD_EXPORT void cupti_error_print_exit(CUptiResult status,
                                        int line,
                                        const char* file,
                                        const char* expr) {
  if (status != CUPTI_SUCCESS) {
    const char* error_string = NULL;
    
    cuptiGetResultString(status, &error_string);
      
    printf("FATAL - CUPTI ERROR: %s:%i:'%s' failed. [Reason] %s\n",
           file,
           line,
           expr,
           error_string);
      
    exit(ECUPTI);
  }
}

NVCD_EXPORT void cupti_warn_print(CUptiResult status,
                                  int line,
                                  const char* file,
                                  const char* expr) {
  if (status != CUPTI_SUCCESS) {
    const char* error_string = NULL;
    
    cuptiGetResultString(status, &error_string);
      
    printf("WARNING - CUPTI ERROR: %s:%i:'%s' failed. [Reason] %s\n",
           file,
           line,
           expr,
           error_string);
  }
}


NVCD_EXPORT void assert_impl(bool cond, const char* expr, const char* file, int line) {
  if (!cond) {
    printf("ASSERT failure: \"%s\" @ %s:%i\n", expr, file, line);
    exit(EASSERT);
  }
}

C_LINKAGE_END
