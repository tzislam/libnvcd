#ifndef __CUPTI_LOOKUP_H__
#define __CUPTI_LOOKUP_H__

#include "commondef.h"
#include <cupti.h>

C_LINKAGE_START

#define NUM_CUPTI_EVENTS_2X 71

extern const char* g_cupti_event_names_2x[NUM_CUPTI_EVENTS_2X];

typedef struct cupti_event_data {
	// one large contiguous buffer,
	// for all groups, constant size.
	// Offsets of these are sent to
	// cuptiEventGroupReadAllEvents()
	CUpti_EventID* 	event_id_buffer;
	uint64_t* event_counter_buffer;

	// indexed strictly per group, constant size
	uint32_t* num_events_per_group;
	uint32_t* num_instances_per_group;
	uint32_t* event_counter_buffer_offsets;
	uint32_t* event_id_buffer_offsets;

	// arbitrary, has a max size which can grow
	uint64_t* kernel_times_nsec_start;
	uint64_t* kernel_times_nsec_end;
	
	CUpti_EventGroup* event_groups;
	
	const char** event_names;

	uint64_t stage_time_nsec_start;
	uint64_t stage_time_nsec_end;
	
	CUcontext context;
	
  uint32_t num_events;
	uint32_t num_event_groups;
	uint32_t num_kernel_times;

	uint32_t event_counter_buffer_length;
	uint32_t event_id_buffer_length;
	uint32_t kernel_times_nsec_buffer_length;
	
} cupti_event_data_t;


#define NUM_CUPTI_METRICS_3X 127

extern const char* g_cupti_metrics_3x[NUM_CUPTI_METRICS_3X];

void cupti_map_event_name_to_id(char* event_name, CUpti_EventID event_id);

const char* cupti_find_event_name_from_id(CUpti_EventID id);

void cupti_name_map_free();

C_LINKAGE_END
#endif //__CUPTI_LOOKUP_H__
