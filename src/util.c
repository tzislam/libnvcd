#include "nvcd/util.h"
#include "nvcd/nvcd.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdarg.h>

#define __USE_GNU // for RTLD_NEXT
#include <dlfcn.h>

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

static inline bool msg_ok(msg_level_t m) {
  // TODO:
  // provide a better configuration interface for this.
  const bool opt_verbose_output = false;
  const bool opt_diagnostic_output = false;  
  return
    (m == MSG_LEVEL_VERBOSE && opt_verbose_output == true) ||
    (m == MSG_LEVEL_DIAG && opt_diagnostic_output == true) ||
    (m != MSG_LEVEL_VERBOSE && m != MSG_LEVEL_DIAG);
}

static NVCD_THREAD_LOCAL const size_t MSG_BUFFER_SZ = 1 << 14;
static NVCD_THREAD_LOCAL char* g_msg_buffer = NULL;

void msg_impl(msg_level_t m, int line, const char* file, const char* fn, const char* msg, ...) {
  if (g_msg_buffer == NULL) {
    g_msg_buffer = zallocNN(MSG_BUFFER_SZ);
  }

  if (msg_ok(m)) {  
    memset(g_msg_buffer, 0, MSG_BUFFER_SZ);

#if defined (NVCD_DEBUG)
    char prefix[1024 + 256] = {0};
    char sig[1024] = {0};
#else
    char prefix[256] = {0};
#endif
    
    va_list ap;
    va_start(ap, msg);
    size_t count = (size_t) vsnprintf(g_msg_buffer, MSG_BUFFER_SZ, msg, ap);
    va_end(ap);
  
    strcat(prefix, "[");
    {
      char tmp[64] = {0};
      snprintf(&tmp[0], 64, "thread_id = %" PRIu64, get_thread_id());
      strcat(prefix, tmp);
    }
    strcat(prefix, ",msg_level = ");
    switch (m) {
    case MSG_LEVEL_VERBOSE:
      strcat(prefix, "VERBOSE");
      break;
    case MSG_LEVEL_ERROR:
      strcat(prefix, "ERROR");
      break;
      // no need to burden user with
      // detailed information - if anyone wishes to
      // override this, we can change the behavior quickly.
    case MSG_LEVEL_USER:
      break;
    case MSG_LEVEL_WARNING:
      strcat(prefix, "WARNING");
      break;
    case MSG_LEVEL_DIAG:
      strcat(prefix, "DIAGNOSTIC");
      break;
    }

    strcat(prefix, "]");
  
#if defined(NVCD_DEBUG)
    if (m != MSG_LEVEL_USER) {
      strcat(prefix, "[");
      snprintf(&sig[0], 1024, "%s:%s:%i", fn, file, line);
      strcat(prefix, sig);
      strcat(prefix, "]");
    }
#endif

    if (m != MSG_LEVEL_USER) {
      fprintf(stdout, "%s:%s", prefix, g_msg_buffer);
    } else {
      fprintf(stdout, "%s", g_msg_buffer);
    }

    fflush(stdout);
  }
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
    exit_msg(stdout, EASSERT, "ASSERT failure: %s@%s:%i\n", expr, file, line);
  }
}

NVCD_EXPORT bool c_assert_impl(bool cond, const char* expr, const char* file, int line) {
  IF_ASSERTS_ENABLED(if (!cond) {
    exit_msg(stdout, EASSERT, "C_ASSERT failure: %s@%s:%i\n", expr,file,line);
  });
  return cond;
}

NVCD_EXPORT void write_logf_impl(const char* func,
				 const char* file,
				 int line,
				 const char* message,
				 ...) {  
  size_t len = strlen(message) + 256;
  char* buffer = zallocNN(len * sizeof(char));  
  
  va_list ap;
  va_start(ap, message);
  vsprintf(buffer, message, ap);
  va_end(ap);

  fprintf(stdout, "[%s:%s:%i]: %s\n", func, file, line, buffer);

  free(buffer);
}

C_LINKAGE_END

