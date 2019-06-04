#ifndef __COMMONDEF_H__
#define __COMMONDEF_H__

#define CUDA_RUNTIME_FN(expr) cuda_runtime_error_print_exit(expr, __LINE__, __FILE__, #expr)
#define CUDA_DRIVER_FN(expr) cuda_driver_error_print_exit(expr, __LINE__, __FILE__, #expr)
#define CUPTI_FN(expr) cupti_error_print_exit(expr, __LINE__, __FILE__, #expr)
#define CUPTI_FN_WARN(expr) cupti_warn_print(expr, __LINE__, __FILE__, #expr)

/*
 * NOTE: bool appears to automatically be defined for CUDA;
 * because nvcc proxies through gcc, the C source modules
 * need to have stdbool.h included.
 */

#include <stdint.h>

#if !defined(__CUDACC__) && !defined(__cplusplus)
#include <stdbool.h>
#include <inttypes.h>
#endif

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

#ifdef __GNUC__
#define NVCD_EXPORT __attribute__ ((visibility("default")))
#else
#warning "Only GCC is officially supported."
#define NVCD_EXPORT
#endif

#define ASSERT(cond) assert_impl((cond), #cond, __FILE__, __LINE__)

#define NOT_NULL(p_expr) assert_not_null_impl((p_expr), #p_expr, __FILE__, __LINE__) 

#define ARRAY_LENGTH(x) (sizeof((x)) / sizeof((x)[0]))

#define zallocNN(sz) NOT_NULL(zalloc((sz)))
  
#define mallocNN(sz) NOT_NULL(malloc((sz)))

#define double_buffNN(p, elem_sz, l) NOT_NULL( double_buffer_size((p), (elem_sz), (l)) )

#define MAYBE_GROW_BUFFER_U32_NN(p, length, limit)    \
  do {                                                \
    if ((p) != NULL) {                                \
      if ((length) == (limit)) {                      \
        size_t out = (size_t)(limit);                 \
        p = double_buffNN((p), sizeof((p)[0]), &out); \
        (limit) = (uint32_t)out;                      \
      }                                               \
    }                                                 \
  } while (0)


#define safe_free_v(p) safe_free((void**) &(p))

#define V_UNSET (-1)

static int64_t ASSERT_SIZE_INT64_LONGLONGINT[sizeof(int64_t) == sizeof(long long int) ? 1 : (-1)];

typedef int64_t clock64_t;
typedef uint32_t bool32_t; // alignment and portability

#define __FUNC__ __func__

#define STRFMT_TAB1 "\t"
#define STRFMT_TAB2 "\t\t"
#define STRFMT_TAB3 "\t\t\t"
#define STRFMT_TAB4 "\t\t\t\t"

#define STRFMT_NEWL1 "\n"

#define STRFMT_MEMBER_SEP ","

#define STRFMT_PTR_VALUE(T, ptr) #T " " #ptr " = %p"
#define STRFMT_STR_VALUE(T, p_str) #T " " #p_str " = %s"
#define STRFMT_INT_VALUE(T, v, fmt) #T " " #v " = %" fmt

#define STRFMT_UINT64_VALUE(v) STRFMT_INT_VALUE(uint64_t, v, PRIu64)
#define STRFMT_UINT32_VALUE(v) STRFMT_INT_VALUE(uint32_t, v, PRIu32)
#define STRFMT_INT32_VALUE(v) STRFMT_INT_VALUE(int32_t, v, PRId32)


#define STRFMT_HEX32_VALUE(v) STRFMT_INT_VALUE(unsigned long, v, PRIx32)
#define STRFMT_HEX64_VALUE(v) STRFMT_INT_VALUE(unsigned long long, v, PRIx64)

#define STRFMT_BOOL_STR_VALUE(v) STRFMT_INT_VALUE(bool, v, "s")

#define STRFMT_STRUCT_PTR_BEGIN(T, ptr) STRFMT_PTR_VALUE(T, ptr) ": {"
#define STRFMT_STRUCT_PTR_END(ptr) "} END " #ptr

#define STRVAL_BOOL_STR_VALUE(v) ((v) == 1 ? "true" : ((v) == 0 ? "false" : "GARBAGE"))

enum {
  ENO_ERROR = 0,
  EUNSUPPORTED_EVENTS,
  EBAD_INPUT,
  EASSERT,
  ECUDA_DRIVER,
  ECUDA_RUNTIME,
  ECUPTI,
  ERACE_CONDITION
};

#ifdef __cplusplus
#define C_LINKAGE_START extern "C" {
#define C_LINKAGE_END }
#else
#define C_LINKAGE_START
#define C_LINKAGE_END
#endif // __cplusplus

#endif
