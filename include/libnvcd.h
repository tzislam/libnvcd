#ifndef __LIBNVCD_H__
#define __LIBNVCD_H__

#include <dlfcn.h>
#include <assert.h>

typedef void (*libnvcd_begin_fn_t)(const char*);
typedef void (*libnvcd_end_fn_t)(void);

static libnvcd_begin_fn_t libnvcd_begin = NULL;
static libnvcd_end_fn_t libnvcd_end = NULL;

static inline void libnvcd_load(void) {
  if (libnvcd_begin == NULL) libnvcd_begin = (libnvcd_begin_fn_t) dlsym(RTLD_NEXT, "libnvcd_begin");
  if (libnvcd_end == NULL) libnvcd_end = (libnvcd_end_fn_t) dlsym(RTLD_NEXT, "libnvcd_end");
  
  assert(libnvcd_begin != NULL);
  assert(libnvcd_end != NULL);
}

#endif // __LIBNVCD_H__
