#ifndef __LIBNVCD_H__
#define __LIBNVCD_H__

#include <dlfcn.h>
#include <assert.h>
#include <stdint.h>

typedef void (*libnvcd_begin_fn_t)(const char*);
typedef void (*libnvcd_end_fn_t)(void);
typedef void (*libnvcd_time_fn_t)(uint32_t);
typedef void (*libnvcd_time_report_fn_t)(void);

// these function pointers are dynamically loaded
// from the preloaded hook.
static libnvcd_begin_fn_t libnvcd_begin = NULL;
static libnvcd_end_fn_t libnvcd_end = NULL;
static libnvcd_time_fn_t libnvcd_time = NULL;
static libnvcd_time_report_fn_t libnvcd_time_report = NULL;

// Timeflags: a bitwise OR of any of these 
// can be passed to libnvcd_time() to indicate
// which portions the user wishes to benchmark.
// Note that NVCD_TIMEFLAGS_RUN and NVCD_TIMEFLAGS_KERNEL
// will report (near) exact same times if and only if
// there is a single CUPTI event group. Otherwise,
// NVCD_TIMEFLAGS_RUN will time each repeated invocation
// of the CUDA kernel. 
#define NVCD_TIMEFLAGS_NONE 0
#define NVCD_TIMEFLAGS_REGION (1 << 2)
#define NVCD_TIMEFLAGS_KERNEL (1 << 1)
#define NVCD_TIMEFLAGS_RUN (1 << 0)

// Users may prefer this interface over bitwise ORing the flags above.
enum spec
  {
   NVCD_TIMESPEC_000 = NVCD_TIMEFLAGS_NONE,
   NVCD_TIMESPEC_00R = NVCD_TIMEFLAGS_RUN,
   NVCD_TIMESPEC_0K0 = NVCD_TIMEFLAGS_KERNEL,
   NVCD_TIMESPEC_0KR = NVCD_TIMEFLAGS_KERNEL | NVCD_TIMEFLAGS_RUN,
   NVCD_TIMESPEC_R00 = NVCD_TIMEFLAGS_REGION,
   NVCD_TIMESPEC_R0R = NVCD_TIMEFLAGS_REGION | NVCD_TIMEFLAGS_RUN,
   NVCD_TIMESPEC_RK0 = NVCD_TIMEFLAGS_REGION | NVCD_TIMEFLAGS_KERNEL,
   NVCD_TIMESPEC_RKR = NVCD_TIMEFLAGS_REGION | NVCD_TIMEFLAGS_KERNEL | NVCD_TIMEFLAGS_RUN       
  };

static inline const char* libnvcd_time_str(uint32_t flags) {
  // :^)
#define LIBNVCD_TIMESPEC_STR(x) case x: return #x; break
  switch (flags) {
    LIBNVCD_TIMESPEC_STR(NVCD_TIMESPEC_000);
    LIBNVCD_TIMESPEC_STR(NVCD_TIMESPEC_00R);
    LIBNVCD_TIMESPEC_STR(NVCD_TIMESPEC_0K0);
    LIBNVCD_TIMESPEC_STR(NVCD_TIMESPEC_0KR);
    LIBNVCD_TIMESPEC_STR(NVCD_TIMESPEC_R00);
    LIBNVCD_TIMESPEC_STR(NVCD_TIMESPEC_R0R);
    LIBNVCD_TIMESPEC_STR(NVCD_TIMESPEC_RK0);
    LIBNVCD_TIMESPEC_STR(NVCD_TIMESPEC_RKR);
  default:
    break;
  }
  return "<libnvcd_time_str INVALID>";
#undef LIBNVCD_TIMESPEC_STR
} 

static inline void libnvcd_load(void) {
  // :^)
#define LIBNVCD_LOAD_FN(func)\
  if (func == NULL) {\
    func = (func##_fn_t) dlsym(RTLD_NEXT, #func);\
    assert(func != NULL);\
  }                                               

  LIBNVCD_LOAD_FN(libnvcd_begin);
  LIBNVCD_LOAD_FN(libnvcd_end);
  LIBNVCD_LOAD_FN(libnvcd_time);
  LIBNVCD_LOAD_FN(libnvcd_time_report);

#undef LIBNVCD_LOAD_FN
}

#endif // __LIBNVCD_H__
