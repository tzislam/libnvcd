#ifndef __CUPTI_LOOKUP_H__
#define __CUPTI_LOOKUP_H__

#include "commondef.h"
#include <cupti.h>

C_LINKAGE_START

#define NUM_CUPTI_EVENTS_2X 71

extern const char* g_cupti_event_names_2x[NUM_CUPTI_EVENTS_2X];

typedef uint64_t cupti_uint_t;

typedef struct cupti_event_data {
	cupti_uint_t* counter_buffer;
	CUpti_EventGroup* event_groups;
	const char** event_names;
	size_t num_threads;
	size_t num_events;
	size_t num_event_groups;
} cupti_event_data_t;


#define NUM_CUPTI_METRICS_3X 127

extern const char* g_cupti_metrics_3x[NUM_CUPTI_METRICS_3X];

C_LINKAGE_END
#endif //__CUPTI_LOOKUP_H__
