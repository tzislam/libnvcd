
#ifndef __CUPTI_LOOKUP_H__
#define __CUPTI_LOOKUP_H__

#include "nvcd/commondef.h"

#include <cupti.h>
#include <pthread.h>

C_LINKAGE_START

// TODO:
// these should probably just be internal
// to the module and accessed via their
// respective get functions
extern const uint32_t g_cupti_event_names_2x_length;

extern char* g_cupti_event_names_2x[];

NVCD_EXPORT char** cupti_get_event_names();

NVCD_EXPORT uint32_t cupti_get_num_event_names();

typedef struct cupti_metric_data cupti_metric_data_t;

typedef struct cupti_event_data {
  // one large contiguous buffer,
  // for all groups, constant size.
  // Offsets of these are sent to
  // cuptiEventGroupReadAllEvents()
  CUpti_EventID*  event_id_buffer;
  uint64_t* event_counter_buffer;

  // indexed strictly per group, constant size
  uint32_t* num_events_per_group;
  uint32_t* num_events_read_per_group;
  uint32_t* num_instances_per_group;

  uint32_t* event_counter_buffer_offsets;
  uint32_t* event_id_buffer_offsets;
  uint32_t* event_groups_read; // not all event groups can be read simultaneously

  // arbitrary, has a max size which can grow
  uint64_t* kernel_times_nsec;
  
  CUpti_EventGroup* event_groups;
  
  char * const * event_names;

  cupti_metric_data_t* metric_data; // ONLY applies to the root event_data node
  
  uint64_t stage_time_nsec_start;
  
  CUcontext cuda_context;
  CUdevice cuda_device;

  CUpti_SubscriberHandle subscriber;
  
  // for asserting thread relationship
  // consistency
  pthread_t thread_event_data_init;
  pthread_t thread_event_callback;

  
  uint32_t num_event_groups; 
  uint32_t num_kernel_times;

  //
  // event_groups_read length == num_event_groups;
  // once count_event_groups_read == num_event_groups,
  // we've finished a benchmark for one
  // run.
  //
  uint32_t count_event_groups_read;
  
  uint32_t event_counter_buffer_length;
  uint32_t event_id_buffer_length;
  uint32_t kernel_times_nsec_buffer_length;

    
  // may not be the amount of events actually used;
  // dependent on target device/compute capability
  // support.
  uint32_t event_names_buffer_length;

  bool32_t initialized;
  bool32_t is_root;
} cupti_event_data_t;

typedef struct cupti_metric_data {
  CUpti_MetricID* metric_ids;
  CUpti_MetricValue* metric_values;
  cupti_event_data_t* event_data;
  bool32_t* computed;
  CUptiResult* metric_get_value_results;

  uint32_t num_metrics;
  bool32_t initialized;
} cupti_metric_data_t;

#ifndef PTHREAD_INITIALIZER
#define PTHREAD_INITIALIZER (unsigned long)0
#endif

#define CUPTI_EVENT_DATA_NULL {                                         \
  /*.event_id_buffer =*/ NULL,                                          \
    /*.event_counter_buffer =*/ NULL,                                   \
    /*.num_events_per_group =*/ NULL,                                   \
    /*.num_events_read_per_group =*/ NULL,                              \
    /*.num_instances_per_group =*/ NULL,                                \
      /*.event_counter_buffer_offsets =*/ NULL,                         \
    /*.event_id_buffer_offsets =*/ NULL,                                \
    /*.event_groups_read =*/ NULL,                                      \
    /*.kernel_times_nsec =*/ NULL,                                      \
    /*.event_groups =*/ NULL,                                           \
    /*.event_names =*/ NULL,                                            \
      /*.metric_data =*/ NULL,                                          \
    /*.stage_time_nsec_start =*/ 0,                                     \
    /*.cuda_context =*/ NULL,                                           \
    /*.cuda_device =*/ CU_DEVICE_INVALID,                               \
    /*.subscriber =*/ NULL,                                             \
      /*.thread_event_data_init =*/ PTHREAD_INITIALIZER,                     \
      /*.thread_event_callback =*/ PTHREAD_INITIALIZER,                       \
    /*.num_event_groups =*/ 0,                                          \
    /*.num_kernel_times =*/ 0,                                          \
    /*.count_event_groups_read =*/ 0,                                   \
    /*.event_counter_buffer_length =*/ 0,                               \
    /*.event_id_buffer_length =*/ 0,                                    \
    /*.kernel_times_nsec_buffer_length =*/ 10,                          \
    /*.event_names_buffer_length =*/ 0,                                 \
      /*.initialized =*/ false,                                         \
      /*.is_root =*/ false                                              \
    }




// unread -> can be read, unless an attempt to enable the event group
//           leads to a compatibility error with other enabled event groups.
// read   -> transition strictly from unread to read after a successful
//           call to cuptiEventGroupReadAllEvents();
//           cupti_event_data_t::count_event_groups_read is also incremented here.
// dont read -> incompatible with previously enabled groups;
//              dont attempt to enable, and
//              thus dont attempt to read this time.
//              is set back to unread after all enabled and unread groups
//              have been processed, so it can then be enabled on an attempt
//              in the future.
enum {
  CED_EVENT_GROUP_UNREAD = 0,
  CED_EVENT_GROUP_READ = 1, 
  CED_EVENT_GROUP_DONT_READ = 2,
};

#define NUM_CUPTI_METRICS_3X 127

extern const char* g_cupti_metrics_3x[NUM_CUPTI_METRICS_3X];

NVCD_EXPORT void cupti_map_event_name_to_id(const char* event_name, CUpti_EventID event_id);

NVCD_EXPORT const char* cupti_find_event_name_from_id(CUpti_EventID id);

NVCD_EXPORT void cupti_name_map_free();

NVCD_EXPORT void cupti_report_event_data(cupti_event_data_t* e);

NVCD_EXPORT void CUPTIAPI cupti_event_callback(void* userdata,
                                   CUpti_CallbackDomain domain,
                                   CUpti_CallbackId callback_id,
                                   CUpti_CallbackData* callback_info);

NVCD_EXPORT void cupti_event_data_subscribe(cupti_event_data_t* e);

NVCD_EXPORT void cupti_event_data_unsubscribe(cupti_event_data_t* e);

NVCD_EXPORT void cupti_event_data_init_from_ids(cupti_event_data_t* e,
                                                CUpti_EventID* event_ids,
                                                uint32_t num_event_ids);

NVCD_EXPORT void cupti_event_data_init(cupti_event_data_t* e);

NVCD_EXPORT void cupti_event_data_set_null(cupti_event_data_t* e);

NVCD_EXPORT void cupti_event_data_free(cupti_event_data_t* e);

NVCD_EXPORT void cupti_event_data_begin(cupti_event_data_t* e);

NVCD_EXPORT void cupti_event_data_end(cupti_event_data_t* e);

NVCD_EXPORT char* cupti_event_get_name(CUpti_EventID eid);

NVCD_EXPORT CUpti_EventID* cupti_metric_get_event_ids(CUpti_MetricID metric,
                                                      uint32_t* num_events);

NVCD_EXPORT CUpti_MetricID* cupti_metric_get_ids(CUdevice dev,
                                                 uint32_t* num_metrics);

NVCD_EXPORT char* cupti_metric_get_name(CUpti_MetricID metric);

NVCD_EXPORT void cupti_event_data_calc_metrics(cupti_event_data_t* e);

NVCD_EXPORT bool cupti_event_data_callback_finished(cupti_event_data_t* e);

C_LINKAGE_END
#endif //__CUPTI_LOOKUP_H__
