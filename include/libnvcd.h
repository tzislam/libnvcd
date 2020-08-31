#ifndef __LIBNVCD_H__
#define __LIBNVCD_H__

#include <dlfcn.h>
#include <assert.h>
#include <stdint.h>

typedef void (*libnvcd_begin_fn_t)(const char*);
typedef void (*libnvcd_end_fn_t)(void);
typedef void (*libnvcd_time_fn_t)(uint32_t);

static libnvcd_begin_fn_t libnvcd_begin = NULL;
static libnvcd_end_fn_t libnvcd_end = NULL;
static libnvcd_time_fn_t libnvcd_time = NULL;

#define NVCD_TIMEFLAGS_NONE 0
#define NVCD_TIMEFLAGS_REGION (1 << 2)
#define NVCD_TIMEFLAGS_KERNEL (1 << 1)
#define NVCD_TIMEFLAGS_RUN (1 << 0)

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
  if (libnvcd_begin == NULL) libnvcd_begin = (libnvcd_begin_fn_t) dlsym(RTLD_NEXT, "libnvcd_begin");
  if (libnvcd_end == NULL) libnvcd_end = (libnvcd_end_fn_t) dlsym(RTLD_NEXT, "libnvcd_end");
  if (libnvcd_time == NULL) libnvcd_time = (libnvcd_time_fn_t) dlsym(RTLD_NEXT, "libnvcd_time");

  assert(libnvcd_begin != NULL);
  assert(libnvcd_end != NULL);
  assert(libnvcd_time != NULL);
}

#endif // __LIBNVCD_H__
