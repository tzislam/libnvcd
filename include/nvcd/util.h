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

NVCD_EXPORT void free_strlist(char** list, size_t length);

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

NVCD_EXPORT void cupti_warn_print(CUptiResult status,
                                  int line,
                                  const char* file,
                                  const char* expr);

NVCD_EXPORT void assert_impl(bool cond,
                             const char* expr,
                             const char* file,
                             int line);

NVCD_EXPORT void* assert_not_null_impl(void* p, const char* expr, const char* file, int line);

NVCD_EXPORT void write_logf_impl(const char* func,
				 const char* file,
				 int line,
				 const char* message,
				 ...);

typedef enum darray_error
  {
   DARRAY_ERROR_NONE = 0,
   DARRAY_ERROR_ENOMEM,
   DARRAY_ERROR_BAD_ALLOC
  } darray_error_t;

#define DARRAY_INIT {NULL, 0, 0, DARRAY_ERROR_NONE}

#define darray(type, init_sz, max_growth)	\
  typedef struct darray_##type {		\
    type* buf;					\
    size_t sz;					\
    size_t len;					\
    darray_error_t err;				\
  } darray_##type##_t;				\
  static inline bool32_t darray_##type##_ok(darray_##type##_t* arr) {\
    return							 \
      (arr->err == DARRAY_ERROR_NONE) &&			 \
      (arr->buf != NULL) &&					 \
      (arr->sz > 0) &&						 \
      (arr->len <= arr->sz);					 \
  }									\
  static inline bool32_t darray_##type##_clean(darray_##type##_t* arr) { \
    return							     \
      (arr->err == DARRAY_ERROR_NONE) &&			     \
      (arr->buf == NULL) &&					     \
      (arr->sz == 0) &&						     \
      (arr->len == 0);						     \
  }	                                                             \
  static inline void darray_##type##_alloc(darray_##type##_t* arr) {	\
    if (darray_##type##_clean(arr)) {					\
      arr->buf = zalloc(sizeof(type) * (init_sz));			\
      if (arr->buf != NULL) {						\
        arr->sz = (init_sz);						\
      }									\
      else {								\
	arr->err = DARRAY_ERROR_BAD_ALLOC;				\
      }									\
    }									\
    else {								\
      arr->err = DARRAY_ERROR_BAD_ALLOC;				\
    }									\
  }									\
  static inline void darray_##type##_grow(darray_##type##_t* arr, size_t amt) {\
    if (darray_##type##_ok(arr)) {					\
      ASSERT(amt <= (max_growth));					\
      type* newbuf = realloc(arr->buf, (arr->sz + amt) * sizeof(type)); \
      if (newbuf != NULL) {						\
        arr->buf = newbuf;						\
	arr->sz += amt;							\
      }									\
      else {								\
	arr->err = DARRAY_ERROR_ENOMEM;					\
      }									\
    }									\
  }									\
  static inline void darray_##type##_append(darray_##type##_t* arr, type* elem) {\
    if (darray_##type##_ok(arr)) {					\
      if (arr->len == arr->sz) {					\
	darray_##type##_grow(arr, arr->sz);				\
      }									\
      if (darray_##type##_ok(arr)) {					\
	memcpy(&arr->buf[arr->len], elem, sizeof(type));		\
	arr->len++;							\
      }									\
    }									\
  }									\
  static inline void darray_##type##_free(darray_##type##_t* arr) {	\
    if (darray_##type##_ok(arr)) {					\
      free(arr->buf);							\
      arr->buf = NULL;							\
      arr->sz = 0;							\
      arr->len = 0;							\
    }									\
  }									\


C_LINKAGE_END

#endif // __UTIL_H__
