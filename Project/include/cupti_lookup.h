
#ifndef __CUPTI_LOOKUP_H__
#define __CUPTI_LOOKUP_H__

#include "commondef.h"
#include <cupti.h>

C_LINKAGE_START

#define NUM_CUPTI_EVENTS_2X 71

extern const uint32_t g_cupti_event_names_2x_length;

extern const char* g_cupti_event_names_2x[NUM_CUPTI_EVENTS_2X];

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
  uint64_t* kernel_times_nsec_start;
  uint64_t* kernel_times_nsec_end;
  
  CUpti_EventGroup* event_groups;
  
  const char** event_names;

  uint64_t stage_time_nsec_start;
  uint64_t stage_time_nsec_end;
  
  CUcontext context;
   
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
  
} cupti_event_data_t;

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

void cupti_map_event_name_to_id(const char* event_name, CUpti_EventID event_id);

const char* cupti_find_event_name_from_id(CUpti_EventID id);

void cupti_name_map_free();

void cupti_report_event_data(cupti_event_data_t* e);

C_LINKAGE_END
#endif //__CUPTI_LOOKUP_H__
