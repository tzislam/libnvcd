#ifndef __CUPTI_LOOKUP_H__
#define __CUPTI_LOOKUP_H__

#include "commondef.h"
#include <cupti.h>
#include "list.h"

C_LINKAGE_START

#define NUM_CUPTI_EVENTS_2X 71

extern const char* g_cupti_event_names_2x[NUM_CUPTI_EVENTS_2X];

typedef uint64_t cupti_uint_t;
typedef int16_t cupti_index_t;

typedef struct cupti_eventlist_node {
	list_t self;
	CUpti_EventID event_id;
	cupti_index_t event_name_index;
} cupti_elist_node_t;

#define cupti_elist_node_iter(x) list_node(x, cupti_elist_node_t, self)
#define cupti_elist_node_car(x) list_base(x, cupti_elist_node_t, self)

typedef struct cupti_event_data {
	cupti_uint_t* counter_buffer;
	CUpti_EventGroup* event_groups;
	cupti_elist_node_t** event_group_id_lists;
	const char** event_names;
	size_t num_threads;
	size_t num_events;
	size_t num_event_groups;
} cupti_event_data_t;


#define NUM_CUPTI_METRICS_3X 127

extern const char* g_cupti_metrics_3x[NUM_CUPTI_METRICS_3X];

C_LINKAGE_END
#endif //__CUPTI_LOOKUP_H__
