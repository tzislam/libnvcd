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
#define NVCD_TIMEFLAGS_REGION (1 << 0)
#define NVCD_TIMEFLAGS_KERNEL (1 << 1)
#define NVCD_TIMEFLAGS_RUN (1 << 2)

static inline void libnvcd_load(void) {
  if (libnvcd_begin == NULL) libnvcd_begin = (libnvcd_begin_fn_t) dlsym(RTLD_NEXT, "libnvcd_begin");
  if (libnvcd_end == NULL) libnvcd_end = (libnvcd_end_fn_t) dlsym(RTLD_NEXT, "libnvcd_end");
  if (libnvcd_time == NULL) libnvcd_time = (libnvcd_time_fn_t) dlsym(RTLD_NEXT, "libnvcd_time");

  assert(libnvcd_begin != NULL);
  assert(libnvcd_end != NULL);
  assert(libnvcd_time != NULL);
}

#endif // __LIBNVCD_H__
