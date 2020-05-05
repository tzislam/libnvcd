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

//
// @DARRAY():
//
// This macro defines a generic, type safe interface
// over a buffer that grows dynamically at the user's request,
// or when capacity has been exceeded when inserting an element
// to an instance of the buffer through this API.
//
// The implementation is trivial, with the one exception being that
// there is a small set of error constants that's provided alongside this
// API.
//
// At the time of this writing, there are only 3 error values (including
// the success value), and the simplest way to determine if an error has
// occurred is by calling darray_##type##_ok() on the given instance.
//
// Currently, there is no need for anything sophisticated as far as this
// data structure is concerned.
//
// Some other parameters (apart from the type itself) for the macro are provided.
// These are here to make runtime checks easier and to ensure
// nothing out of the ordinary is happening. The goal is predictablity and making
// bugs easier to find. The less one has to use GDB, the better.
//
// @darray_##type##_ok():
//
// The code in this library is simple. Execution paths
// aren't dynamic or unpredictable enough to warrant a NULL check on the array itself.
//
// So, 1) there is really no reason for a segfault to occur,
// and 2) if a segfault DOES occur, then we can debug it easily enough.
//
// Bottom line: if NULL is actually passed to any one of these routines,
// it's a problem and it needs to be detected.
//
// If, for some reason, external
// third party code is interacting with this directly, or we have some other reason to
// provide a null check, then we can change direction.
//
// @darray_##type##_free():
//
// The reader might be wondering why we don't simply ensure that buf != NULL
// before choosing to free. This is again part of the choice for correctness:
// it is the author's opinion that it's much better in this case to have memory
// leaked because it is more likely to be available in the event of a core dump
//
#define DARRAY(type, init_sz, max_growth)	\
  typedef struct darray_##type {		\
    type* buf;					\
    size_t sz;					\
    size_t len;					\
    darray_error_t err;				\
  } darray_##type##_t;				\
  static inline bool32_t darray_##type##_ok(darray_##type##_t* arr) {\
    ASSERT(arr != NULL);					     \
    return							     \
      (arr->err == DARRAY_ERROR_NONE) &&			     \
      (arr->buf != NULL) &&					     \
      (arr->sz > 0) &&						     \
      (arr->len <= arr->sz);						\
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
  static inline void darray_##type##_appendp(darray_##type##_t* arr, type* elem) {\
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
    static inline void darray_##type##_append(darray_##type##_t* arr, type elem) {\
    if (darray_##type##_ok(arr)) {					\
      if (arr->len == arr->sz) {					\
	darray_##type##_grow(arr, arr->sz);				\
      }									\
      if (darray_##type##_ok(arr)) {					\
	memcpy(&arr->buf[arr->len], &elem, sizeof(type));		\
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
  static inline type darray_##type##_at(darray_##type##_t* arr, size_t i) { \
    ASSERT(i < arr->len);						\
    type r;								\
    if (darray_##type##_ok(arr) && i < arr->len) {			\
      r = arr->buf[i];							\
    }									\
    return r;								\
  }									\
  static inline size_t darray_##type##_size(darray_##type##_t* arr) {	\
    size_t i = UINT64_MAX;						\
    if (darray_##type##_ok(arr)) {					\
      i = arr->len;							\
    }									\
    return i;								\
  }									\
  static inline size_t darray_##type##_capacity(darray_##type##_t* arr) { \
    size_t i = UINT64_MAX;						\
    if (darray_##type##_ok(arr)) {					\
      i = arr->sz;							\
    }									\
    return i;								\
  }									\
  static inline type* darray_##type##_data(darray_##type##_t* arr) {	\
    type* i = NULL;							\
    if (darray_##type##_ok(arr)) {					\
      i = arr->buf;							\
    }									\
    return i;								\
  }									
  
  




C_LINKAGE_END

#endif // __UTIL_H__
