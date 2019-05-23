#include "commondef.h"
#include "gpu.h"
#include "cupti_lookup.h"
#include "list.h"

#include <ctype.h>
#include <cupti.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>

#include <inttypes.h>
#include <pthread.h>

static pthread_t _thread_main;
static pthread_t _thread_callback;

#ifndef ENV_PREFIX
#define ENV_PREFIX "BENCH_"
#endif

#define ENV_EVENTS ENV_PREFIX "EVENTS"

#define ENV_DELIM ':'
#define ENV_ALL_EVENTS "ALL"

/*
 * cupti event
 */ 

static CUpti_runtime_api_trace_cbid g_cupti_runtime_cbids[] = {
  CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
  CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
};

#define NUM_CUPTI_RUNTIME_CBIDS (sizeof(g_cupti_runtime_cbids) / sizeof(g_cupti_runtime_cbids[0]))

static cupti_event_data_t g_event_data = {
  .event_id_buffer = NULL, 
  .event_counter_buffer =NULL, 

  .num_events_per_group = NULL, 
  .num_events_read_per_group = NULL,
  .num_instances_per_group = NULL,

  .event_counter_buffer_offsets = NULL,
  .event_id_buffer_offsets = NULL,
  .event_groups_read = NULL,

  .kernel_times_nsec_start = NULL,
  .kernel_times_nsec_end = NULL,

  .event_groups = NULL,

  .event_names = NULL,

  .stage_time_nsec_start = 0,
  .stage_time_nsec_end = 0,

  .context = NULL,

  .num_event_groups = 0,
  .num_kernel_times = 0,

  .count_event_groups_read = 0,
  
  .event_counter_buffer_length = 0,
  .event_id_buffer_length = 0,
  .kernel_times_nsec_buffer_length = 10, // default; will increase as necessary at runtime

  .event_names_buffer_length = 0
};

static CUpti_SubscriberHandle g_cupti_subscriber = NULL;

void collect_group_events(cupti_event_data_t* e) {
  for (uint32_t i = 0; i < e->num_event_groups; ++i) {
    if (e->event_groups_read[i] == CED_EVENT_GROUP_UNREAD) {
      
      size_t cb_size =
        e->num_events_per_group[i] *
        e->num_instances_per_group[i] *
        sizeof(uint64_t);
    
      size_t cb_offset = e->event_counter_buffer_offsets[i];

      size_t ib_size = e->num_events_per_group[i] * sizeof(CUpti_EventID);
      size_t ib_offset = e->event_id_buffer_offsets[i];

      size_t ids_read = 0;
    
      CUPTI_FN(cuptiEventGroupReadAllEvents(e->event_groups[i],
                                            CUPTI_EVENT_READ_FLAG_NONE,
                                            &cb_size,
                                            &e->event_counter_buffer[cb_offset],
                                            &ib_size,
                                            &e->event_id_buffer[ib_offset],
                                            &ids_read));

      printf("[%i] ids read: %" PRId64 "/ %" PRId64 "\n",
             i,
             ids_read,
             (size_t) e->num_events_per_group[i]);
    }
  }

  for (uint32_t i = 0; i < e->num_event_groups; ++i) {
    if (e->event_groups_read[i] == CED_EVENT_GROUP_UNREAD) {
      e->event_groups_read[i] = CED_EVENT_GROUP_READ;
      e->count_event_groups_read++;
      
      CUPTI_FN(cuptiEventGroupDisable(e->event_groups[i]));
    }
  }

  for (uint32_t i = 0; i < e->num_event_groups; ++i) {
    if (e->event_groups_read[i] == CED_EVENT_GROUP_DONT_READ) {
      e->event_groups_read[i] = CED_EVENT_GROUP_UNREAD;
    }
  }
}

static bool _message_reported = false;

void CUPTIAPI cupti_event_callback(void* userdata,
                                   CUpti_CallbackDomain domain,
                                   CUpti_CallbackId callback_id,
                                   CUpti_CallbackData* callback_info) {
  {
    bool found = false;
    size_t i = 0;

    while (i < NUM_CUPTI_RUNTIME_CBIDS && !found) {
      found = callback_id == g_cupti_runtime_cbids[i];
      i++;
    }

    ASSERT(found);
  }

  // For now it appears that the threads are the same between the main thread
  // and the thread this callback is installed in. The check is important though
  // since this could technically change. Some might consider this pedantic, but non-thread-safe
  // event handlers with user pointer data are a thing, and device synchronization waits
  // can obviously happen across multiple threads.
  {
    _thread_callback = pthread_self();
    
    volatile int threads_eq = pthread_equal(_thread_callback, _thread_main);

    if (threads_eq != 0) {
      if (!_message_reported) {
        fprintf(stderr, "%s is launched on the same thread as the main thread (this is good)\n", __FUNC__);
        _message_reported = true;
      }
    } else {
      exit_msg(stderr,
               ERACE_CONDITION,
               "Race condition detected in %s. "
               "Synchronization primitives will be needed for "
               "main thread wait loop and this callback\n", __FUNC__);
    }
  }
  {
    cupti_event_data_t* event_data = (cupti_event_data_t*) userdata;
    
    switch (callback_info->callbackSite) {
    case CUPTI_API_ENTER: {
      CUDA_RUNTIME_FN(cudaDeviceSynchronize());

      CUPTI_FN(cuptiSetEventCollectionMode(callback_info->context,
                                           CUPTI_EVENT_COLLECTION_MODE_KERNEL));

      //
      // We can get all of the event groups we wish to read,
      // but not necessarily at the same time.
      // In this case, it's necessary to repeatedly call the same kernel
      // until
      //           event_data->count_event_groups_read == event_data->num_event_groups
      // is true.
      // The state tracking is handled in this loop,
      // as well as in collect_group_events()
      //
      for (uint32_t i = 0; i < event_data->num_event_groups; ++i) {
        if (event_data->event_groups_read[i] == CED_EVENT_GROUP_UNREAD) {
          CUptiResult err = cuptiEventGroupEnable(event_data->event_groups[i]);

          if (err != CUPTI_SUCCESS) {
            if (err == CUPTI_ERROR_NOT_COMPATIBLE) {
              printf("Group %" PRIu32 " out of "
                     "%" PRIu32 " considered not compatible with the current set of enabled groups\n",
                     i,
                     event_data->num_event_groups);

              event_data->event_groups_read[i] = CED_EVENT_GROUP_DONT_READ;
            } else {
              CUPTI_FN(err);
            }
          } else {
            printf("Group %" PRIu32 " enabled.\n", i);
          }
        }
      }

      CUPTI_FN(cuptiDeviceGetTimestamp(callback_info->context,
                                       &event_data->stage_time_nsec_start));
    } break;

    case CUPTI_API_EXIT: {
      CUDA_RUNTIME_FN(cudaDeviceSynchronize());

      CUPTI_FN(cuptiDeviceGetTimestamp(callback_info->context,
                                       &event_data->stage_time_nsec_end));
      
      collect_group_events(event_data);     
      
    } break;

    default:
      ASSERT(false);
      break;
    }
  }
}

void cupti_subscribe() {
  CUPTI_FN(cuptiSubscribe(&g_cupti_subscriber,
                          (CUpti_CallbackFunc)cupti_event_callback,
                          &g_event_data));

  for (uint32_t i = 0; i < NUM_CUPTI_RUNTIME_CBIDS; ++i) {
    CUPTI_FN(cuptiEnableCallback(1,
                                 g_cupti_subscriber,
                                 CUPTI_CB_DOMAIN_RUNTIME_API,
                                 g_cupti_runtime_cbids[i]));
  }
}

void cupti_unsubscribe() {
  CUPTI_FN(cuptiUnsubscribe(g_cupti_subscriber));
}

void init_cupti_event_groups(CUcontext ctx,
                             CUdevice dev,
                             cupti_event_data_t* e) {
#define MAX_EGS 30
  // static default; increase if more groups become necessary
  uint32_t max_egs = MAX_EGS; 
  uint32_t num_egs = 0;

  // we use a local buffer with an estimate,
  // so when we store the memory we aren't using
  // more than we need
  CUpti_EventGroup local_eg_assign[MAX_EGS];

  // CUpti_EventGroup is just a typedef for a pointer
  for (uint32_t i = 0; i < max_egs; ++i)
    local_eg_assign[i] = NULL;
    
#undef MAX_EGS
  
  for (uint32_t i = 0; i < e->event_names_buffer_length; ++i) {
    CUpti_EventID event_id = V_UNSET;
    
    CUptiResult err = cuptiEventGetIdFromName(dev,
                                              e->event_names[i],
                                              &event_id);

    // even if the compute capability being targeted
    // is technically larger than the capability of the
    // set of events queried against, there is still variation between
    // cards. Some events simply won't be available for that
    // card.
    bool available = true;
    
    if (err != CUPTI_SUCCESS) {
      if (err == CUPTI_ERROR_INVALID_EVENT_NAME) {
        available = false;
      } else {
        // force an exit, since
        // something else needs to be
        // looked at
        CUPTI_FN(err);
      }
    } else {
      // in the future we'll only have ids,
      // so we may as well map them now for output.
      cupti_map_event_name_to_id(e->event_names[i], event_id);
    }
    
    uint32_t event_group = 0;
    
    if (available) {
      uint32_t j = 0;
      err = CUPTI_ERROR_NOT_COMPATIBLE;

      //
      // find a suitable group
      // for this event
      //
      bool iterating = j < max_egs;
      bool error_valid = false;

      while (iterating) {
        if (local_eg_assign[j] == NULL) {
          CUPTI_FN(cuptiEventGroupCreate(ctx,
                                         &local_eg_assign[j],
                                         0));
          num_egs++;
        }

        err = cuptiEventGroupAddEvent(local_eg_assign[j],
                                      event_id);
        
        event_group = j;
        j++;

        // event groups cannot have
        // events from different domains;
        // in these cases we just find another group.
        error_valid =
          !(err == CUPTI_ERROR_MAX_LIMIT_REACHED
            || err == CUPTI_ERROR_NOT_COMPATIBLE);

        if (error_valid) {
          error_valid = err == CUPTI_SUCCESS;
        }
        
        if (j == max_egs || error_valid) {
          iterating = false;
        }
      }

      ASSERT(j <= max_egs);
      
      // trigger exit if we still error out:
      // something not taken into account
      // needs to be looked at
      CUPTI_FN(err);
    }

    printf("(%s) index %u, group_index %u => %s:0x%x\n",
           available ? "available" : "unavailable",
           i,
           event_group,
           e->event_names[i],
           event_id);
  }

  ASSERT(num_egs <= max_egs /* see the declaration of max_egs if this fails */);

  if (num_egs == 0) {
    exit_msg(stderr,
             EUNSUPPORTED_EVENTS,
             "%s",
             "No supported events found within given list. "
             "Support can vary between device and compute capability.");
  }
  
  // fill our event groups buffer
  {
    e->num_event_groups = num_egs;
    e->event_groups = zallocNN(sizeof(e->event_groups[0]) * e->num_event_groups);
    e->event_groups_read = zallocNN(sizeof(e->event_groups_read[0]) * e->num_event_groups);
    
    for (uint32_t i = 0; i < e->num_event_groups; ++i) {
      ASSERT(local_eg_assign[i] != NULL);
      
      e->event_groups[i] = local_eg_assign[i];
    }
  }
}

void init_cupti_event_data(CUcontext ctx,
                           CUdevice dev,
                           cupti_event_data_t* e) {
  ASSERT(ctx != NULL);
  ASSERT(dev >= 0);

  init_cupti_event_groups(ctx, dev, e);

  // get instance and event counts for each group
  {
    e->num_events_per_group = zallocNN(sizeof(e->num_events_per_group[0]) *
                                       e->num_event_groups);

    e->num_events_read_per_group = zallocNN(sizeof(e->num_events_read_per_group[0]) *
                                            e->num_event_groups);
    
    e->num_instances_per_group = zallocNN(sizeof(e->num_instances_per_group[0]) *
                                          e->num_event_groups);

    for (uint32_t i = 0; i < e->num_event_groups; ++i) {
      // instance count
      {
        uint32_t inst = 0;
        size_t instsz = sizeof(inst);
        CUPTI_FN(cuptiEventGroupGetAttribute(e->event_groups[i],
                                             CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                             &instsz,
                                             &inst));

        e->num_instances_per_group[i] = inst;
      }

      // event count
      {
        uint32_t event = 0;
        size_t eventsz = sizeof(event);
        CUPTI_FN(cuptiEventGroupGetAttribute(e->event_groups[i],
                                             CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                             &eventsz,
                                             &event));
        e->num_events_per_group[i] = event;
      }
    }                                          
  }
  
  // compute offsets for the event id buffer,
  // and allocate the memory.
  // for all groups
  {
    e->event_id_buffer_offsets = mallocNN(sizeof(e->event_id_buffer_offsets[0]) *
                                          e->num_event_groups);
    
    e->event_id_buffer_length = 0;
    
    for (uint32_t i = 0; i < e->num_event_groups; ++i) {
      e->event_id_buffer_offsets[i] = e->event_id_buffer_length;
      e->event_id_buffer_length += e->num_events_per_group[i];
    }

    e->event_id_buffer = zallocNN(sizeof(e->event_id_buffer[0]) *
                                  e->event_id_buffer_length);
    
  }
  
  // compute offset indices for the event counter buffer,
  // and allocate the memory.
  // for all groups
  {
    e->event_counter_buffer_offsets =
      mallocNN(sizeof(e->event_counter_buffer_offsets[0]) * e->num_event_groups);
    
    e->event_counter_buffer_length = 0;
    
    for (uint32_t i = 0; i < e->num_event_groups; ++i) {
      uint32_t accum = 0;
      for (uint32_t j = 0; j < i; ++j) {
        accum += e->num_events_per_group[j] * e->num_instances_per_group[j];
      }
      
      e->event_counter_buffer_offsets[i] = accum;
    }
    
    for (uint32_t i = 0; i < e->num_event_groups; ++i) {
      e->event_counter_buffer_length +=
        e->num_events_per_group[i] * e->num_instances_per_group[i];
    }
    
    e->event_counter_buffer =
      zallocNN(sizeof(e->event_counter_buffer[0]) * e->event_counter_buffer_length);  
  } 

}

void free_cupti_event_data(cupti_event_data_t* e) {
  ASSERT(e != NULL);
  
  safe_free_v(e->event_id_buffer);
  safe_free_v(e->event_counter_buffer);
  
  safe_free_v(e->num_events_per_group);
  safe_free_v(e->num_instances_per_group);
  safe_free_v(e->num_events_read_per_group);
  
  safe_free_v(e->event_counter_buffer_offsets);
  safe_free_v(e->event_id_buffer_offsets);
  safe_free_v(e->event_groups_read);

  safe_free_v(e->kernel_times_nsec_start);
  safe_free_v(e->kernel_times_nsec_end);
  
  for (size_t i = 0; i < e->num_event_groups; ++i) { 
    if (e->event_groups[i] != NULL) {
      CUPTI_FN(cuptiEventGroupRemoveAllEvents(e->event_groups[i]));
      CUPTI_FN(cuptiEventGroupDestroy(e->event_groups[i]));
    }
  }
  
  safe_free_v(e->event_groups);
  
  // TODO: event names may be either a subset of a static buffer
  // initialized in the .data section,
  // or a subset. Should add a flag to determine
  // whether or not the data needs to be freed.
  memset(e, 0, sizeof(*e));
}

/*
 * CUDA
 */

static CUdevice g_cuda_device = CU_DEVICE_INVALID;
static CUcontext g_cuda_context = NULL;

CUdevice cuda_get_device() {
  ASSERT(g_cuda_device != CU_DEVICE_INVALID);
  return g_cuda_device;
}

void cuda_set_device(CUdevice dev) {
  g_cuda_device = dev;
}

CUcontext cuda_get_context() {  
  if (g_cuda_context == NULL) {
    CUDA_DRIVER_FN(cuCtxCreate(&g_cuda_context, 0, cuda_get_device()));
  }
  
  return g_cuda_context;
}

void free_cuda_data() {
  ASSERT(g_cuda_context != NULL);
  ASSERT(g_cuda_device != CU_DEVICE_INVALID);

  CUDA_DRIVER_FN(cuCtxDestroy(g_cuda_context));
}

void free_cupti_data() {
  free_cupti_event_data(&g_event_data);
  cupti_name_map_free();
}

/*
 * env var list parsing
 *
 */

const char* env_var_list_start(const char* list) {
  const char* p = list;

  while (*p && *p != '=') {
    p++;
  }

  ASSERT(*p == '=');

  const char* ret = p + 1;

  if (!isalpha(*ret)) {
    printf("ERROR: %s must begin with a letter.\n", ret);
    ret = NULL;
  }

  return ret;
}

const char* env_var_list_scan_entry(const char* p, size_t* p_count) {
  size_t count = 0;

  bool error = false;
  
  while (*p && *p != ENV_DELIM && !error) {
    error = !isalnum(*p) && !(*p == '_');
    
    if (error) {
      printf("ERROR: invalid character found: %s.\n", p);
    } else {
      count++;
      p++;
    }
  }

  if (p_count != NULL) {
    *p_count = count;
  }

  if (error) {
    p = NULL;
  }

  return p;
}

typedef int (*env_var_list_scan_fn_t)(const char* entry, size_t entry_len, void* user);

typedef void (*env_var_list_scan_error_fn_t)(void* user);

struct env_var_list_scan_ctx {
  char** list;
  size_t index;
  size_t num_elems;
};

int env_var_list_count_entry(const char* entry, size_t entry_len, void* user) {
  ASSERT(user != NULL);
  size_t* count = (size_t*) user;
  *count = *count + 1;
  return 1;
}

void env_var_list_count_entry_error(void* user) {
  ASSERT(user != NULL);

  size_t* count = (size_t*) user;
  *count = 0;
}

int env_var_list_insert_entry(const char* entry, size_t entry_len, void* user) {
  ASSERT(user != NULL);
  struct env_var_list_scan_ctx* ctx = (struct env_var_list_scan_ctx*) user;
  
  char* str = zalloc((entry_len + 1) * sizeof(char));

  ASSERT(ctx->index < ctx->num_elems);
  
  if (str != NULL) {
    strncpy(str, entry, entry_len);

    ctx->list[ctx->index] = str;
    ctx->index++;
  }

  return str != NULL;
}

void env_var_list_scan(const char* var,
                       env_var_list_scan_fn_t callback,
                       env_var_list_scan_error_fn_t error,
                       void* user) {
  const char* p = var;

  if (p != NULL) {
    const char* delim = strchr(p, ENV_DELIM);

    bool scanning = *p != '\0';
    
    while (scanning) {
      size_t this_count = 0;

      if (env_var_list_scan_entry(p, &this_count) == NULL) {
        scanning = false;
      } else {
        scanning = this_count != 0;
      }
    
      if (scanning) {
        scanning = callback(p, this_count, user);
        if (scanning) {
          if (delim != NULL) {
            p = delim + 1;
            delim = strchr(p, ENV_DELIM);
          } else {
            scanning = false;
          }
        }
      } else if (error != NULL) {
        error(user);
      }
    }
  }
}

char** env_var_list_read(const char* env_var_value, size_t* count) {
  struct env_var_list_scan_ctx ctx = { 0 };
  
  env_var_list_scan(env_var_value,
                    env_var_list_count_entry,
                    env_var_list_count_entry_error,
                    &ctx.num_elems);
  
  if (ctx.num_elems) { 
    ctx.list = zalloc(ctx.num_elems * sizeof(char*));
  }
  
  if (ctx.list != NULL) {
    env_var_list_scan(env_var_value,
                      env_var_list_insert_entry,
                      NULL,
                      &ctx);
  }

  if (count != NULL) {
    *count = ctx.num_elems;
  }

  return ctx.list;
}

/*
 * env var list testing
 *
 */

struct test {
  uint8_t print_info;
  uint8_t run;
} static g_test_params = {
  false,
  false
};

void test_env_var(char* str, size_t expected_count, bool should_null) {
  if (g_test_params.print_info) {
    printf("Testing %s. Expecting %s with a count of %lu\n",
           str,
           should_null ? "failure" : "success",
           expected_count);
  }
  
  size_t count = 0;
  char** list = env_var_list_read(str, &count);

  if (should_null) {
    ASSERT(list == NULL);
    ASSERT(count == 0);

    if (g_test_params.print_info) {
      printf("env_var_list_read for %s returned NULL\n", str);
    }
  } else {
    ASSERT(count == expected_count);
    ASSERT(list != NULL);

    for (size_t i = 0; i < count; ++i) {
      printf("[%lu]: %s\n", i, list[i]);
    }

    for (size_t i = 0; i < count; ++i) {
      ASSERT(list[i] != NULL);
      free(list[i]);
    }

    free(list);
  }
}

void test_env_parse() {
  test_env_var("BLANK=::", 0, 1);
  test_env_var("VALID=this:is:a:set:of:strings", 6, 0);
  test_env_var("MALFORMED=this::is:a::bad:string", 0, 1);
}

#define PTIME_FMT "f"

typedef double profile_time_t;

profile_time_t profile_time() {
  struct timeval t = {0};
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

typedef char string_micro_t[64];
 
typedef struct profile_data { 
  string_micro_t* device_names;
  CUdevice* device_ids; 
  
  profile_time_t start;
  profile_time_t time; /* TODO: add darray since time slices will need to be taken */

  int num_devices;
  int device; /* Defaults to zero */
  
  uint8_t needs_init;
} profile_data_t;

static profile_data_t* g_data = NULL;

void free_profile_data(profile_data_t* data) {
  free(data->device_names);
  free(data->device_ids);
  free(data);
}
 
profile_data_t* default_profile_data() {
  if (g_data == NULL) {
    {
      g_data = zalloc(sizeof(*g_data));
      ASSERT(g_data != NULL);

      g_data->needs_init = true;
    }
  }

  return g_data;
}

void profile_data_print(profile_data_t* data) {
  for (int i = 0; i < data->num_devices; ++i) {
    printf("device: %i. device id: 0x%x. name: \"%s\"\n",
           i,
           data->device_ids[i],
           data->device_names[i]);
  }
}

void build_event_list(cupti_event_data_t* e) {
  char* env_string = getenv(ENV_EVENTS);

  FILE* stream = stderr;
  
  if (env_string != NULL) { 
    size_t count = 0;
    char* const* list = env_var_list_read(env_string, &count);

    // Sanity check
    ASSERT(count < 0x1FF);

    if (list != NULL) {
      size_t i = 0;

      bool scanning = i < count;
      bool using_all = false;
    
      while (scanning) {
        if (strcmp(list[i], ENV_ALL_EVENTS) == 0) {
          fprintf(stream,
            "(%s) Found %s in list. All event counters will be used.\n",
            ENV_EVENTS,
            ENV_ALL_EVENTS);
          
          e->event_names = &g_cupti_event_names_2x[0];
          e->event_names_buffer_length = g_cupti_event_names_2x_length;

          scanning = false;
          using_all = true;
        } else {
          fprintf(stream, "(%s) [%" PRIu64 "] Found %s\n", ENV_EVENTS, i, list[i]);
          i++;
          scanning = i < count;
        }
      }

      if (!using_all) {
        e->event_names = list;
        e->event_names_buffer_length = (uint32_t)count;
      }
    } else {
      exit_msg(stream,
               EBAD_INPUT,
               "(%s) %s",
               ENV_EVENTS,
               "Could not parse the environment list.");
    }
  } else {
    fprintf(stream,
            "%s undefined; defaulting to all event counters.\n",
            ENV_EVENTS);
    
    e->event_names = &g_cupti_event_names_2x[0];
    e->event_names_buffer_length = g_cupti_event_names_2x_length;
  }
}


void cupti_benchmark_start(cupti_event_data_t* event_data) {
  ASSERT(event_data != NULL);

  build_event_list(event_data);
  
  profile_data_t* data = default_profile_data();

  if (data->needs_init) {
    CUDA_DRIVER_FN(cuInit(0));

    {
      CUDA_RUNTIME_FN(cudaGetDeviceCount(&data->num_devices));

      data->device_ids =
        zallocNN(sizeof(*(data->device_ids)) * data->num_devices);

      data->device_names =
        zallocNN(sizeof(*(data->device_names)) * data->num_devices);
    
      for (int i = 0; i < data->num_devices; ++i) {
        CUDA_DRIVER_FN(cuDeviceGet(&data->device_ids[i], i));
      
        CUDA_DRIVER_FN(cuDeviceGetName(data->device_names[i],
                                       sizeof(data->device_names[i]) - 1,
                                       data->device_ids[i]));
      }

      cuda_set_device(data->device_ids[0]);
    }

    init_cupti_event_data(cuda_get_context(), cuda_get_device(), event_data);

    cupti_subscribe();
    
    profile_data_print(data);
    
    data->needs_init = false;
  }

  data->start = profile_time();
}

void cupti_benchmark_end() {
}

void cleanup() {
  free_profile_data(default_profile_data());
  free_cupti_data();
  free_cuda_data();
  cupti_unsubscribe();
}

int main() {
  if (g_test_params.run) {
    test_env_parse();
  }

  (void)g_cupti_subscriber;
  (void)g_cupti_runtime_cbids;

  _thread_main = pthread_self();

  int threads = 1024;
  
  cupti_benchmark_start(&g_event_data);
  
  clock64_t* thread_times = zallocNN(sizeof(thread_times[0]) * threads);

  while (g_event_data.count_event_groups_read
         < g_event_data.num_event_groups) {
    
    gpu_test_matrix_vec_mul(threads, thread_times);
    CUDA_RUNTIME_FN(cudaDeviceSynchronize());
  }
  
  for (int i = 0; i < threads; ++i) {
    printf("[%i] time: %llu\n", i, thread_times[i]);
  }

  cupti_report_event_data(&g_event_data);
  
  cupti_benchmark_end();

  cleanup();
  
  return 0;
}
