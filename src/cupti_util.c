#include "nvcd/cupti_util.h"
#include <string.h>
#include <inttypes.h>
#include "nvcd/util.h"
#include "nvcd/env_var.h"

#define MAX_EVENT_GROUPS_PER_EVENT_DATA 250

typedef CUpti_EventID cupti_event_id;
DARRAY(cupti_event_id, 128, 128);

#define PRINT_GROUP_ATTR_SCALAR(group, type, enum_value)	do {	\
    type v##enum_value = 0;						\
  size_t sz = sizeof(v##enum_value);					\
  CUPTI_FN(cuptiEventGroupGetAttribute(group,				\
                                       enum_value,			\
                                       &sz,				\
                                       &(v##enum_value)));		\
  msg_verbosef(STRFMT_TAB1 STRFMT_UINT32_VALUE(v##enum_value) STRFMT_NEWL1,\
	       (v##enum_value));					\
  } while (0)

static void print_event_group_attr_info(CUpti_EventGroup group, const char* opt_tag) {
  if (group != NULL) {
    msg_verbose_begin();
    
    if (opt_tag != NULL) {
      msg_verbosef("[%s] ", opt_tag);
    }
    
    msg_verbosef("CUPTI attributes for group %p {\n", group);

    PRINT_GROUP_ATTR_SCALAR(group,
			    CUpti_EventDomainID,
			    CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID);
    PRINT_GROUP_ATTR_SCALAR(group,
			    int32_t,
			    CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES);
    PRINT_GROUP_ATTR_SCALAR(group,
			    uint32_t,
			    CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS);   

    // enumerate events:
    {
      uint32_t event_count = 0;
      size_t sz_count = sizeof(event_count);
      CUPTI_FN(cuptiEventGroupGetAttribute(group,
					   CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
					   &sz_count,
					   &event_count));
      
      size_t sz_buf = sizeof(CUpti_EventID) * event_count;
      CUpti_EventID* event_ids = zallocNN(sz_buf);
      CUPTI_FN(cuptiEventGroupGetAttribute(group,
					   CUPTI_EVENT_GROUP_ATTR_EVENTS,
					   &sz_buf,
					   event_ids));

      msg_verbosef(STRFMT_TAB1 STRFMT_STRUCT_PTR_BEGIN(CUpti_EventID*, event_ids) STRFMT_NEWL1,
	     event_ids);
      for (uint32_t i = 0; i < event_count; ++i) {
	
	const char* n = cupti_event_get_name(event_ids[i]);
	  
        msg_verbosef("\t\tevent_ids[%" PRIu32 "] = %" PRIu32 ", %s" STRFMT_NEWL1,
		     i, event_ids[i], n);
	
      }
      msg_verboses(STRFMT_TAB1 STRFMT_STRUCT_PTR_END(event_ids) STRFMT_NEWL1);

      free(event_ids);
    }

    PRINT_GROUP_ATTR_SCALAR(group,
			    uint32_t,
			    CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT);

    msg_verbose_end();
  }
}

static darray_cupti_event_id_t* query_event_list(cupti_event_data_t* e) {
  ASSERT(e != NULL);
  
  
  // in the event that there is a problem,
  // NULL is returned.
  darray_cupti_event_id_t* ret = NULL;
  
  
  char** event_names = NULL;
  uint32_t num_event_names = 0;
  
  CUpti_EventDomainID* domain_buffer = NULL;
  size_t domain_buffer_size = 0;
  uint32_t num_event_domains = 0;
  
  CUPTI_FN(cuptiDeviceGetNumEventDomains(e->cuda_device, &num_event_domains));
  ASSERT(num_event_domains != 0);

  domain_buffer_size = sizeof(domain_buffer[0]) * (size_t)num_event_domains;
  domain_buffer = malloc(domain_buffer_size);

  if (domain_buffer != NULL) {
    CUPTI_FN(cuptiDeviceEnumEventDomains(e->cuda_device, &domain_buffer_size, &domain_buffer[0]));

    // if everything is OK at the end of this routine,
    // we set 'ret' to this pointer.
    
    darray_cupti_event_id_t* event_id_list = zallocNN(sizeof(*event_id_list));
    darray_cupti_event_id_alloc(event_id_list);
    
    if (darray_cupti_event_id_ok(event_id_list)) {
      bool32_t ok = true;
      uint32_t i = 0;
      while (ok && i < num_event_domains) {
	CUpti_EventID* event_buffer = NULL;
	size_t event_buffer_size = 0;
	uint32_t num_events = 0;

	CUPTI_FN(cuptiEventDomainGetNumEvents(domain_buffer[i], &num_events));
	ASSERT(num_events != 0);

	event_buffer_size = sizeof(event_buffer[0]) * (size_t)num_events;

	// there isn't a simple means of preallocating everything up front,
	// so we check every iteration if we need to grow to add more
	// events to the list
	if ((darray_cupti_event_id_size(event_id_list) + num_events) >=
	    darray_cupti_event_id_capacity(event_id_list)) {
	  
	  darray_cupti_event_id_grow(event_id_list, num_events << 4);
	}

	ok = darray_cupti_event_id_ok(event_id_list);

	// a call to grow may have failed, so we just
	// check every iteration
	if (ok) {       	
	  CUPTI_FN(cuptiEventDomainEnumEvents(domain_buffer[i],
					      &event_buffer_size,
					      darray_cupti_event_id_data(event_id_list) +
					      darray_cupti_event_id_size(event_id_list)
					      ));

	  event_id_list->len += num_events;
	  num_event_names += num_events;

	  i++;
	}
      }

      if (ok) {
	ret = event_id_list;
      }
    } // darray_cupti_event_id_ok
  }

  return ret;
}

NVCD_EXPORT char** cupti_get_event_names(cupti_event_data_t* e, size_t* out_len) {  
  char** event_names = NULL;
  
  darray_cupti_event_id_t* event_id_list = query_event_list(e);
  
  if (darray_cupti_event_id_ok(event_id_list)) {    
    event_names = mallocNN(darray_cupti_event_id_size(event_id_list) * sizeof(char*));	
    for (uint32_t i = 0; i < darray_cupti_event_id_size(event_id_list); ++i) {
      event_names[i] = cupti_event_get_name(darray_cupti_event_id_data(event_id_list)[i]);
    }

    IF_NN_THEN(out_len,
	       *out_len = darray_cupti_event_id_size(event_id_list));	       
  }

  darray_cupti_event_id_free(event_id_list);

  return event_names;
}

NVCD_EXPORT uint32_t cupti_get_num_event_names(cupti_event_data_t* e) {
  uint32_t ret = UINT32_MAX;

  darray_cupti_event_id_t* event_id_list = query_event_list(e);
  
  if (darray_cupti_event_id_ok(event_id_list)) {
    ret = (uint32_t) darray_cupti_event_id_size(event_id_list);
  }

  darray_cupti_event_id_free(event_id_list);
  
  return ret;
}

static CUpti_runtime_api_trace_cbid g_cupti_runtime_cbids[] = {
  CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
  CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
};

#define NUM_CUPTI_RUNTIME_CBIDS (sizeof(g_cupti_runtime_cbids) / sizeof(g_cupti_runtime_cbids[0]))

static void init_cupti_event_buffers(cupti_event_data_t* e);

static void fill_event_groups(cupti_event_data_t* e,
                              CUpti_EventGroup* local_eg_assign,
                              uint32_t num_egs) {
  e->num_event_groups = num_egs;
  e->event_groups = zallocNN(sizeof(e->event_groups[0]) * e->num_event_groups);
  e->event_group_read_states = zallocNN(sizeof(e->event_group_read_states[0]) * e->num_event_groups);
  e->event_groups_enabled = zallocNN(sizeof(e->event_groups_enabled[0]) * e->num_event_groups);
  
  for (uint32_t i = 0; i < e->num_event_groups; ++i) {
    ASSERT(local_eg_assign[i] != NULL);
      
    e->event_groups[i] = local_eg_assign[i];

    // enabling this (on the GTX 960M, at least) will cause a SIGBUS error for event groups
    // that are tied to events which are used by metrics
#if 0
    uint32_t profile_all = 1;
    CUPTI_FN(cuptiEventGroupSetAttribute(e->event_groups[i],
                                         CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                         sizeof(profile_all), &profile_all));
#endif
  }
}

static bool find_event_group(cupti_event_data_t* e,
                             CUpti_EventGroup* local_eg_assign,
                             CUpti_EventID event_id,
                             uint32_t max_egs,
                             uint32_t* num_egs) {
  
  uint32_t j = 0;
  CUptiResult err = CUPTI_ERROR_NOT_COMPATIBLE;

  //
  // find a suitable group
  // for this event
  //
  bool iterating = j < max_egs;
  bool error_valid = false;
  bool found = false;

  while (iterating) {
    if (local_eg_assign[j] == NULL) {
      CUPTI_FN(cuptiEventGroupCreate(e->cuda_context,
                                     &local_eg_assign[j],
                                     0));
      *num_egs = *num_egs + 1;
    }

    err = cuptiEventGroupAddEvent(local_eg_assign[j],
                                  event_id);
        
    j++;

    // event groups cannot have
    // events from different domains;
    // in these cases we just find another group.
    // see https://docs.nvidia.com/cuda/cupti/group__CUPTI__EVENT__API.html#group__CUPTI__EVENT__API_1g649750f363752bccbf9e98582d5f6925
    error_valid =
      !(err == CUPTI_ERROR_MAX_LIMIT_REACHED
        || err == CUPTI_ERROR_NOT_COMPATIBLE);

    if (error_valid) {
      error_valid = err == CUPTI_SUCCESS;
      CUPTI_FN(err);
    }
        
    if (j == max_egs || error_valid) {
      iterating = false;
      found = err == CUPTI_SUCCESS;
    }
  }

  ASSERT(j <= max_egs);
      
  // trigger exit if we still error out:
  // something not taken into account
  // needs to be looked at
  CUPTI_FN(err);

  return found;
}

static CUpti_MetricID* fetch_metric_ids_from_device(CUdevice device, uint32_t* num_metrics) {

  msg_verboses("Fetching all metric IDs from device");
  
  CUPTI_FN(cuptiDeviceGetNumMetrics(device, num_metrics));

  size_t metric_array_size = sizeof(CUpti_MetricID) * (*num_metrics);

  CUpti_MetricID* out_ids = zallocNN(metric_array_size);
  
  CUPTI_FN(cuptiDeviceEnumMetrics(device,
                                  &metric_array_size,
                                  &out_ids[0]));

  return out_ids;
}

static CUpti_MetricID* fetch_metric_ids_from_names(CUdevice device,
                                                   char** metric_names,
                                                   uint32_t* num_metrics) {

  msg_verbose_begin();
  
  msg_verboses("Attempting to fetch metric id values from");
  for (uint32_t i = 0; i < *num_metrics; ++i) {
    msg_verbosef("%s: ", metric_names[i]);
  }
  msg_verboseline();
  
  uint32_t desired = *num_metrics;
  
  size_t metric_array_size = sizeof(CUpti_MetricID) * (*num_metrics);

  CUpti_MetricID* out_ids = zallocNN(metric_array_size);

  uint32_t i = 0;
  uint32_t j = 0;
  
  while (i < *num_metrics && j < desired) {
    CUpti_MetricID id;
    CUptiResult err = cuptiMetricGetIdFromName(device,
					       metric_names[j],
					       &id);

    if (err != CUPTI_SUCCESS) {
      msg_verbosef("fetch_metric_ids_from_names: Could not find metric name \'%s\'\n",
		   metric_names[j]);
      
      *num_metrics = *num_metrics - 1;
    } else {
      msg_verbosef("fetch_metric_ids_from_names: Found ID %" PRIx32 " for metric \'%s\'\n",
		   id,
		   metric_names[j]);
      out_ids[i] = id;
      i++;
    }

    j++;

    CUPTI_FN_WARN(err);
  }

  ASSERT(j == desired && i == *num_metrics);

  msg_verbosef("Found %" PRIu32 " / %" PRIu32 " metrics.\n", *num_metrics, desired);

  if (*num_metrics == 0) {
    free(out_ids);
    out_ids = NULL;
  } else if (*num_metrics != desired) {
    size_t sz = sizeof(CUpti_MetricID) * (*num_metrics);
    CUpti_MetricID* new_ids = zallocNN(sz);
    memcpy(new_ids, out_ids, sz);
    free(out_ids);
    out_ids = new_ids;
  }

  msg_verbose_end();

  return out_ids;
}

static CUpti_MetricID* fetch_metric_ids(CUdevice device,
                                        uint32_t* num_metrics) {
  char* metric_env_value = getenv(ENV_METRICS);

  CUpti_MetricID* buf = NULL;
  
  if (metric_env_value != NULL) {
    size_t count = 0;

    char** list = env_var_list_read(metric_env_value,
                                    &count);

    if (list != NULL) {
      ASSERT(count > 0);
        
      *num_metrics = (uint32_t) count;

      buf = fetch_metric_ids_from_names(device,
                                        list,
                                        num_metrics);
    } else {
      ASSERT(count == 0);
      
      buf = fetch_metric_ids_from_device(device,
                                         num_metrics);
    }
  } else {
    buf = fetch_metric_ids_from_device(device,
                                       num_metrics);
  }

  return buf;
}

static void init_cupti_metric_data(cupti_event_data_t* e) {
  cupti_metric_data_t* metric_buffer = zallocNN(sizeof(*metric_buffer));
  
  metric_buffer->metric_ids = fetch_metric_ids(e->cuda_device, &metric_buffer->num_metrics);

  metric_buffer->metric_values = zallocNN(sizeof(metric_buffer->metric_values[0]) *
                                          metric_buffer->num_metrics);
  
  metric_buffer->event_data = zallocNN(sizeof(metric_buffer->event_data[0]) *
                                       metric_buffer->num_metrics);

  metric_buffer->computed = zallocNN(sizeof(metric_buffer->computed[0]) *
                                     metric_buffer->num_metrics);

  metric_buffer->metric_get_value_results =
    zallocNN(sizeof(metric_buffer->metric_get_value_results[0]) *
             metric_buffer->num_metrics);
                                                     
#define _index_ "[%" PRIu32 "] "
  msg_verbose_begin();
  for (uint32_t i = 0; i < metric_buffer->num_metrics; ++i) {
    char* name = cupti_metric_get_name(metric_buffer->metric_ids[i]);
    msg_verbosef( _index_ "Processing metric %s...\n", i, name);
    
    uint32_t num_events = 0;
    CUPTI_FN(cuptiMetricGetNumEvents(metric_buffer->metric_ids[i], &num_events));
    msg_verbosef(_index_ "event count is %" PRIu32 "\n", i, num_events);

    size_t event_array_size = sizeof(CUpti_EventID) * num_events;
    
    CUpti_EventID* events = zallocNN(event_array_size);
    
    CUPTI_FN(cuptiMetricEnumEvents(metric_buffer->metric_ids[i],
                                   &event_array_size,
                                   &events[0]));
    
    cupti_event_data_set_null(&metric_buffer->event_data[i]);

    metric_buffer->event_data[i].cuda_device = e->cuda_device;
    metric_buffer->event_data[i].cuda_context = e->cuda_context;
    
    cupti_event_data_init_from_ids(&metric_buffer->event_data[i],
                                   &events[0],
                                   num_events);
    
    // TODO: take each group and compute the needed values...
    
    free(name);

    msg_verboses("---");
  }
  msg_verbose_end();
  
  metric_buffer->initialized = true;

  e->metric_data = metric_buffer;

}

static uint32_t derive_event_count(cupti_event_data_t* e) {
  uint32_t ret = 0;

  for (uint32_t i = 0; i < e->num_event_groups; ++i) {
    ret += e->num_events_per_group[i];
  }

  return ret;
}

//
// see https://docs.nvidia.com/cuda/archive/9.2/cupti/group__CUPTI__METRIC__API.html#group__CUPTI__METRIC__API_1gf42dcb1d349f91265e7809bbba2fc01e
// for more information on why this is needed.
//
static void normalize_counters(cupti_event_data_t* e, uint64_t* normalized) {
  for (uint32_t j = 0; j < e->num_event_groups; ++j) {
    CUpti_EventDomainID domain_id;
    size_t domain_id_sz = sizeof(domain_id);
    CUPTI_FN(cuptiEventGroupGetAttribute(e->event_groups[j],
                                         CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                                         &domain_id_sz,
                                         (void*) &domain_id));
    
    uint32_t total_instance_count = 1;

    // Querying for this value will throw
    // an error that isn't consistent with what's reported
    // in the documentation. Currently, only speculation
    // as to "why" is possible - for now, we're forgoing
    // anything involving multiple domain instances
    // since it appears to cause problems with specific configurations
    // (at least, for the metrics - event counters that are recorded without
    // interfacing with teh metrics appear to be fine)
    // Note that if the reader tries this on their machine,
    // it may work fine. This was tested with cuda 9.2 on a GTX 960 M.
#if 0    
    size_t total_instance_count_sz = sizeof(total_instance_count);
    CUPTI_FN(cuptiEventDomainGetAttribute(domain_id,
                                          CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                                          &total_instance_count_sz,
                                          (void*) &total_instance_count));
#endif
    
    uint32_t cb_offset = e->event_counter_buffer_offsets[j];
    uint32_t ib_offset = e->event_id_buffer_offsets[j];
    uint32_t nepg = e->num_events_per_group[j];
    uint32_t nipg = e->num_instances_per_group[j];
      
    for (uint32_t row = 0; row < nipg; ++row) {
      for (uint32_t col = 0; col < nepg; ++col) {
        normalized[ib_offset + col] +=
          e->event_counter_buffer[cb_offset + row * nepg + col];
      }
    }

    for (uint32_t k = 0; k < nepg; ++k) {
      normalized[ib_offset + k] *= total_instance_count;
      normalized[ib_offset + k] /= nipg;
    }
  }
}

static void print_cupti_metric(cupti_metric_data_t* metric_data, uint32_t index) {
  //ASSERT(metric_data->computed[index] == true);

  CUpti_MetricID m = metric_data->metric_ids[index];
  CUpti_MetricValue v = metric_data->metric_values[index];

  char* name = cupti_metric_get_name(m);
    
  msg_userf(METRICS_TAG "index[%" PRIu32 "] %s = ", index, name);
  
  if (metric_data->computed[index] == true) {
  
    CUpti_MetricValueKind kind;
  
    size_t kind_sz = sizeof(kind);
  
    CUPTI_FN(cuptiMetricGetAttribute(m,
                                     CUPTI_METRIC_ATTR_VALUE_KIND,
                                     &kind_sz,
                                     (void*) &kind));
  
    switch (kind) {
    case CUPTI_METRIC_VALUE_KIND_DOUBLE: {
      msg_userf("(double) %f", v.metricValueDouble);
    } break;
    
    case CUPTI_METRIC_VALUE_KIND_UINT64: {
      msg_userf("(uint64) %" PRIu64, v.metricValueUint64);
    } break;
    
    case CUPTI_METRIC_VALUE_KIND_INT64: {
      msg_userf("(int64) %" PRId64, v.metricValueInt64);
    } break;
    
    case CUPTI_METRIC_VALUE_KIND_PERCENT: {
      msg_userf("(percent) %f", v.metricValuePercent);
    } break;
    
    case CUPTI_METRIC_VALUE_KIND_THROUGHPUT: {
      msg_userf("(bytes/second) %" PRId64, v.metricValueThroughput);
    } break;
    
    case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL: {
      const char* level = NULL;
    
      switch (v.metricValueUtilizationLevel) {
      case CUPTI_METRIC_VALUE_UTILIZATION_IDLE: { 
        level = "IDLE";
      } break;
      
      case CUPTI_METRIC_VALUE_UTILIZATION_LOW: { 
        level = "LOW";
      } break;

      case CUPTI_METRIC_VALUE_UTILIZATION_MID: { 
        level = "MID";
      } break;

      case CUPTI_METRIC_VALUE_UTILIZATION_HIGH: { 
        level = "HIGH";
      } break;

      case CUPTI_METRIC_VALUE_UTILIZATION_MAX: { 
        level = "MAX";
      } break;
      
      default:
        //        ASSERT(false /* bad utilization level received */);

        level = "UNKNOWN (bad value received)";
        break;
      }
    
      msg_userf("(utilization level) %" PRIu32 " =  %s ",
		v.metricValueUtilizationLevel,
		level);
    } break;

    default:
      //ASSERT(false /* bad metric value kind received */);
      msg_warnf(METRICS_TAG "bad metric value kind received: 0x%" PRIx32 " ", kind);
      break;
    }
  } else {
    const char* result_string = NULL;

    // TODO: docs do not specify whether or not this string should be freed,
    // so the assumption is that it shouldn't (for now - should test with valgrind
    // when time allows)
    //
    // reference (see function listing given below the enum values):
    //
    // https://docs.nvidia.com/cuda/archive/9.2/cupti/group__CUPTI__RESULT__API.html#group__CUPTI__RESULT__API_1g8c54bf95108e67d858f37fcf76c88714
    //
    //
    CUPTI_FN(cuptiGetResultString(metric_data->metric_get_value_results[index],
				  &result_string));
    
    msg_warnf(METRICS_TAG "metric NOT computed - Error code received: %" PRIu32 " = %s",
	      metric_data->metric_get_value_results[index],
	      result_string);
  }
  
  msg_userline();

  free(name);
}

static void calc_cupti_metrics(cupti_metric_data_t* m) {
  ASSERT(m->initialized == true);
  ASSERT(m->num_metrics < 2000);
  
  for (uint32_t i = 0; i < m->num_metrics; ++i) {
    cupti_event_data_t* e = &m->event_data[i];
    
    ASSERT(e->event_id_buffer_length ==
           derive_event_count(&m->event_data[i]));

    ASSERT(e->num_kernel_times > 0 && e->num_kernel_times < 100);
    
    uint64_t* normalized = zallocNN(e->event_id_buffer_length *
                                    sizeof(normalized[0]));
    
    normalize_counters(e, normalized);
    
    CUptiResult err = cuptiMetricGetValue(e->cuda_device,
                                          m->metric_ids[i],
                                          sizeof(e->event_id_buffer[0]) * e->event_id_buffer_length,
                                          &e->event_id_buffer[0],
                                          sizeof(normalized[0]) * e->event_id_buffer_length,
                                          &normalized[0],
                                          e->kernel_times_nsec[0],
                                          &m->metric_values[i]);

    ASSERT(m->computed[i] == false);

    if (err == CUPTI_SUCCESS) {
      m->computed[i] = true;
    } else if (err != CUPTI_ERROR_INVALID_METRIC_VALUE) {
      msg_warnf(METRICS_TAG "error for metric %" PRIu32 " = 0x%" PRIx32 "\n", i, m->metric_ids[i]);
      for (uint32_t j = 0; j < e->event_id_buffer_length; ++j) {
        msg_warnf(METRICS_TAG "\tEvent ID %" PRIu32 " = 0x%" PRIx32 "\n", j, e->event_id_buffer[j]);
      }
      CUPTI_FN_WARN(err);
    }

    m->metric_get_value_results[i] = err;
  }
}


typedef struct group_info {
  uint32_t num_events;
  uint32_t num_instances;
  CUpti_EventID* events;
  uint64_t* counters;
  CUpti_EventGroup group;
} group_info_t; 

static volatile bool g_process_group_aos = false;

static group_info_t* g_group_info_buffer = NULL;
static uint32_t g_group_info_count = 0;
static uint32_t g_group_info_size = 0;

static void group_info_append(group_info_t* info, uint32_t group) {
  if (g_group_info_buffer == NULL) {
    g_group_info_size = 50;
    g_group_info_buffer = zallocNN(g_group_info_size * sizeof(*g_group_info_buffer));
  }

  ASSERT(group < g_group_info_size /* TODO: use double_buffer_size() in util.h */); 
  
  memcpy(&g_group_info_buffer[group], info, sizeof(*info));
  g_group_info_count++;
}

static void group_info_free(group_info_t* info) {
  safe_free_v(info->events);
  safe_free_v(info->counters);
}

static void group_info_validate(cupti_event_data_t* e,
                                group_info_t* info,
                                uint32_t group) {
  msg_verbosef("Validating group %" PRIu32 "\n", group);

  ASSERT(info->group == e->event_groups[group]);
  ASSERT(info->num_events == e->num_events_per_group[group]);
  ASSERT(info->num_instances == e->num_instances_per_group[group]);

  volatile CUpti_EventID* base =
    &e->event_id_buffer[e->event_id_buffer_offsets[group]];
  
  for (uint32_t i = 0; i < info->num_events; ++i) {
    ASSERT(base[i] == info->events[i]);

    char* name = cupti_event_get_name(info->events[i]);
    char* name2 = cupti_event_get_name(base[i]);

    ASSERT(strcmp(name, name2) == 0);

#if 0
    uint64_t* soa_counters = &e->event_counter_buffer[e->event_counter_buffer_offsets[group]];
    
    // counter comparison
    for (uint32_t j = 0; j < info->num_instances; ++j) {
      ASSERT(soa_counters[i * info->num_instances + j] ==
             info->counters[i * info->num_instances + j]);
    }
#endif
    
    msg_verbosef("\t[%" PRIu32 "] %s|%s is good...\n", i, name, name2);
    free(name);
    free(name2);
  }

  ASSERT(info->num_instances == 1);
}

static void group_info_make(cupti_event_data_t* e,
			    group_info_t* info,
			    uint32_t group) {

  CUpti_EventGroup g = e->event_groups[group];
  
  {
    size_t sz = sizeof(info->num_events);

    CUPTI_FN(cuptiEventGroupGetAttribute(g,
                                         CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                         &sz,
                                         &info->num_events));
  }

  {
    size_t sz = sizeof(info->num_instances);

    CUPTI_FN(cuptiEventGroupGetAttribute(g,
                                         CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                         &sz,
                                         &info->num_instances));
  }

  {
    size_t sz = sizeof(info->events[0]) * info->num_events;

    info->events = zallocNN(sz);
    
    CUPTI_FN(cuptiEventGroupGetAttribute(g,
                                         CUPTI_EVENT_GROUP_ATTR_EVENTS,
                                         &sz,
                                         &info->events[0]));
  }
  
  {
    size_t sz = sizeof(info->counters) * info->num_instances * info->num_events;
    info->counters = zallocNN(sz);
  }

  info->group = g;
}

static group_info_t* group_info_new(cupti_event_data_t* e, uint32_t group) {
  group_info_t* ret = zallocNN(sizeof(*ret));
  group_info_make(e, ret, group);
  return ret;
}

static void read_group_all_events(cupti_event_data_t* e, uint32_t group) {
  size_t cb_size =
    e->num_events_per_group[group] *
    e->num_instances_per_group[group] *
    sizeof(uint64_t);

  //  uint64_t* tmp = zallocNN(cb_size);
  
  {
    size_t ib_size = e->num_events_per_group[group] * sizeof(CUpti_EventID);
    size_t ib_offset = e->event_id_buffer_offsets[group];

    size_t ids_read = 0;

    size_t cb_offset = e->event_counter_buffer_offsets[group];
    
    CUPTI_FN(cuptiEventGroupReadAllEvents(e->event_groups[group],
                                          CUPTI_EVENT_READ_FLAG_NONE,
                                          &cb_size,
                                          &e->event_counter_buffer[cb_offset],
                                          &ib_size,
                                          &e->event_id_buffer[ib_offset],
                                          &ids_read));

    msg_verbosef("group[%i] ids read: %" PRId64 "/ %" PRId64 "\n",
		 group,
		 ids_read,
		 (size_t) e->num_events_per_group[group]);
  }
}

// if called, it must be called after read_group_all_events
static void read_group_per_event(cupti_event_data_t* e, uint32_t group) {  
  group_info_t* info = group_info_new(e, group);

  size_t bytes_read_per_event = sizeof(info->counters[0]) * info->num_instances;
  
  for (uint32_t i = 0; i < info->num_events; ++i) {
    CUPTI_FN(cuptiEventGroupReadEvent(info->group,
                                      CUPTI_EVENT_READ_FLAG_NONE,
                                      info->events[i],
                                      &bytes_read_per_event,
                                      &info->counters[i * info->num_instances]));

    ASSERT(bytes_read_per_event == sizeof(info->counters[0]) * info->num_instances);
  }

  for (uint32_t i = 0; i < g_group_info_count; ++i) {
    ASSERT(info->group != g_group_info_buffer[i].group);
  }
  
  group_info_append(info, group);
}

static void init_cupti_event_groups(cupti_event_data_t* e) {
  msg_verbosef("%s\n", "init_cupti_event_groups_entered");
  
  // static default; increase if more groups become necessary
  uint32_t max_egs = MAX_EVENT_GROUPS_PER_EVENT_DATA; 
  uint32_t num_egs = 0;

  // we use a local buffer with an estimate,
  // so when we store the memory we aren't using
  // more than we need
  CUpti_EventGroup local_eg_assign[MAX_EVENT_GROUPS_PER_EVENT_DATA];

  // CUpti_EventGroup is just a typedef for a pointer
  for (uint32_t i = 0; i < max_egs; ++i)
    local_eg_assign[i] = NULL;
  
  for (uint32_t i = 0; i < e->event_names_buffer_length; ++i) {
    CUpti_EventID event_id = V_UNSET;

    msg_verbosef(
            "Attempting to find ID for event device %" PRId32
            "; event [%" PRId32"] = %s\n", e->cuda_device,
            i, e->event_names[i]);
    
    CUptiResult err = cuptiEventGetIdFromName(e->cuda_device,
                                              e->event_names[i],
                                              &event_id);

    //-------------------------------------------------
    // FIXME(?): this routine was written when a static list of counters
    // was being used for testing. Now that we're querying for them directly through
    // the device/driver itself, it's questionable how much the following
    // error checks are needed.
    //
    // Still, for now, this code is harmless, and will at least act as
    // a buffer for anything that's been overlooked.
    //-------------------------------------------------
    
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
    }
    
    if (available) {
      bool found = find_event_group(e,
                                    &local_eg_assign[0],
                                    event_id,
                                    max_egs,
                                    &num_egs);
      ASSERT(found);
    }

    msg_verbosef("(%s) group found for index %u => %s:0x%x\n",
		 available ? "available" : "unavailable",
		 i,
		 e->event_names[i],
		 event_id);
  }

  ASSERT(num_egs <= max_egs /* see the declaration of max_egs if this fails */);

  if (num_egs == 0) {
    exit_msg(stdout,
             EUNSUPPORTED_EVENTS,
             "%s",
             "No supported events found within given list. "
             "Support can vary between device and compute capability.");
  }

  fill_event_groups(e, &local_eg_assign[0], num_egs);
}

static void init_cupti_event_names(cupti_event_data_t* e) {
  char* env_string = getenv(ENV_EVENTS);

  FILE* stream = stderr;
  
  if (env_string != NULL) { 
    size_t count = 0;
    char** list = env_var_list_read(env_string, &count);

    // Sanity check
    ASSERT(count < 0x1FF);

    if (list != NULL) {
      size_t i = 0;

      bool scanning = i < count;
      bool using_all = false;
    
      while (scanning) {
        if (strcmp(list[i], ENV_ALL_EVENTS) == 0) {
          msg_verbosef("(%s) Found %s in list. All event counters will be used.\n",
		       ENV_EVENTS,
		       ENV_ALL_EVENTS);

	  size_t num_event_names = 0;
          e->event_names = cupti_get_event_names(e, &num_event_names);
	  e->event_names_buffer_length = (uint32_t)num_event_names;

          scanning = false;
          using_all = true;
        } else {
          msg_verbosef("(%s) [%" PRIu64 "] Found %s\n", ENV_EVENTS, i, list[i]);
          i++;
          scanning = i < count;
        }
      }

      if (!using_all) {
        msg_verbosef("(%s) NOT USING ALL\n", ENV_EVENTS);
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
    msg_verbosef("%s undefined; defaulting to all event counters.\n",
		 ENV_EVENTS);
    
    size_t num_event_names = 0;
    e->event_names = cupti_get_event_names(e, &num_event_names);
    e->event_names_buffer_length = (uint32_t)num_event_names;
  }
}

static void init_cupti_event_buffers(cupti_event_data_t* e) {
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

    // is meant to be written to after the counter measurements
    // have been taken.
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

static const size_t PEG_BUFFER_SZ = 1 << 20;
static char* _peg_buffer = NULL;

static void print_event_group_soa(cupti_event_data_t* e, uint32_t group) {
  if (_peg_buffer == NULL) {
    _peg_buffer = zallocNN(PEG_BUFFER_SZ);
  }
  // used for iterative bounds checking
#define peg_buffer_length PEG_BUFFER_SZ - 1
  
  memset(&_peg_buffer[0], 0, PEG_BUFFER_SZ);
  
  uint64_t* pcounters = &e->event_counter_buffer[0];
  
  uint32_t ib_offset = e->event_id_buffer_offsets[group];
  uint32_t cb_offset = e->event_counter_buffer_offsets[group];
  
  uint32_t nepg = e->num_events_per_group[group];
  uint32_t nipg = e->num_instances_per_group[group];

  IF_ASSERTS_ENABLED(volatile uint32_t next_cb_offset = 0;
		     volatile uint32_t next_ib_offset = 0;);


  int ptr = 0;
  
  IF_ASSERTS_ENABLED({
      // bounds check ordering for
      // event_counter_buffer_offsets      
      volatile uint32_t prev_cb_offset = (group > 0) ?
      
	e->event_counter_buffer_offsets[group - 1] :
	0;

      volatile uint32_t prev_cb_offset_add = (group > 0) ?

	(e->num_events_per_group[group - 1] *
	 e->num_instances_per_group[group - 1]) :
	0;

      ASSERT(prev_cb_offset + prev_cb_offset_add == cb_offset);
    }

    {    
      // bounds check ordering for
      // event_id_buffer_offsets
      volatile uint32_t prev_ib_offset = (group > 0) ?

	e->event_id_buffer_offsets[group - 1] :
	0;

      volatile uint32_t prev_ib_offset_add = (group > 0) ?
      
	e->num_events_per_group[group - 1] :
	0;

      ASSERT(prev_ib_offset + prev_ib_offset_add == ib_offset);
    }

    {
      // used for iterative bounds checking
      next_cb_offset =
	group < (e->num_event_groups - 1) ?
	e->event_counter_buffer_offsets[group + 1] :
	e->event_counter_buffer_length;
    }
  
    {
      // used for iterative bounds checking
      next_ib_offset =
	group < (e->num_event_groups - 1) ?
	e->event_id_buffer_offsets[group + 1] :
	e->event_id_buffer_length;
    });
  
  for (uint32_t i = 0; i < nepg; ++i) {
    ASSERT(ib_offset + i < next_ib_offset);
    ASSERT(ptr < peg_buffer_length);

    CUpti_EventID eid = e->event_id_buffer[ib_offset + i];
    
    {
      char* name = cupti_event_get_name(eid);
      
      ptr += sprintf(&_peg_buffer[ptr],
                     "event[%" PRIu32 "](id = 0x%" PRIx32 ", name = %s)\n",
                     i,
                     eid,
                     name);

      free(name);
    }
    
    for (uint32_t j = 0; j < nipg; ++j) {
      uint32_t k = cb_offset + j * nepg + i;

      ASSERT(k < next_cb_offset);
      
      ptr += sprintf(&_peg_buffer[ptr],
                     "\n\tevent_instance_counter[%" PRIu32 "] = %" PRIu64 "\n\n",
                     j,
                     pcounters[k]);
    }
  }

  ASSERT(ptr <= peg_buffer_length);
  
  msg_verbosef("====== GROUP %" PRIu32  "=======\n"
		"%s"
		"===\n",
		group,
		&_peg_buffer[0]);

#undef peg_buffer_length
}

static void print_event_group_aos(cupti_event_data_t* e, uint32_t group) {
  ASSERT(group < g_group_info_count);
  
  group_info_t* info = &g_group_info_buffer[group];
  
  msg_verbosef("======(AOS) GROUP %" PRIu32  "=======\n", group);
  
  for (uint32_t i = 0; i < info->num_events; ++i){
    char* name = cupti_event_get_name(info->events[i]);

    msg_verbosef("[%" PRIu32 "] %s\n", i, name);
    
    for (uint32_t j = 0; j < info->num_instances; ++j) {
      msg_verbosef("\t[%" PRIu32 "] %" PRIu64 "\n",
		   j,
		   info->counters[i * info->num_instances + j]);
    }

    free(name);
  }

  msg_verboses("===");
}

NVCD_EXPORT void cupti_report_event_data(cupti_event_data_t* e) {
  if (g_process_group_aos) {
    for (uint32_t i = 0; i < e->num_event_groups; ++i) {
      group_info_t* info = &g_group_info_buffer[i];
      group_info_validate(e, info, i);
    }
  }
  
  for (uint32_t i = 0; i < e->num_event_groups; ++i) {
    print_event_group_soa(e, i);
    if (g_process_group_aos) {
      print_event_group_aos(e, i);
    }
  }

  if (e->is_root == true) {
    ASSERT(e->metric_data != NULL);
    
    for (uint32_t i = 0; i < e->metric_data->num_metrics; ++i) {
      print_cupti_metric(e->metric_data, i);
    }    
  }
}

static void collect_group_events(cupti_event_data_t* e) {
  for (uint32_t i = 0; i < e->num_event_groups; ++i) {
    if (e->event_group_read_states[i] == CED_EVENT_GROUP_UNREAD) {
      read_group_all_events(e, i);

      if (g_process_group_aos) {
        read_group_per_event(e, i);
        group_info_validate(e,
                            &g_group_info_buffer[i],
                            i);
      }
    }
  }

  // errornous groups;
  // These were never enabled, and the errors they triggered
  // don't have anything to do with compatibility with other enabled groups.
  // So, they don't need to be disabled by cupti. Increment is still
  // necessary for the host to be aware that profiling the kernel is finished.
  for (uint32_t i = 0; i < e->num_event_groups; ++i) {
    if (e->event_group_read_states[i] == CED_EVENT_GROUP_SKIP) {
      e->event_group_read_states[i] = CED_EVENT_GROUP_READ;
      e->count_event_groups_read++;     
    }
  }


  // Groups which we've read for this particular callback instance
  for (uint32_t i = 0; i < e->num_event_groups; ++i) {
    if (e->event_group_read_states[i] == CED_EVENT_GROUP_UNREAD) {
      e->event_group_read_states[i] = CED_EVENT_GROUP_READ;
      e->count_event_groups_read++;

      e->event_groups_enabled[i] = false;
      CUPTI_FN(cuptiEventGroupDisable(e->event_groups[i]));
    }
  }

  // Groups which we haven't read yet, but weren't compatible
  // with the ones already enabled
  for (uint32_t i = 0; i < e->num_event_groups; ++i) {
    if (e->event_group_read_states[i] == CED_EVENT_GROUP_DONT_READ) {
      e->event_group_read_states[i] = CED_EVENT_GROUP_UNREAD;
    }
  }
}


static bool _message_reported = false;

NVCD_EXPORT void CUPTIAPI cupti_event_callback(void* userdata,
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

  cupti_event_data_t* event_data = (cupti_event_data_t*) userdata;

  msg_verboses("-----------------");
  msg_verbosef("event callback hit for event_data = %p\n", event_data);
  
  // For now it appears that the threads are the same between the main thread
  // and the thread this callback is installed in. The check is important though
  // since this could technically change. Some might consider this pedantic, but non-thread-safe
  // event handlers with user pointer data are a thing, and device synchronization waits
  // can obviously happen across multiple threads.  
  {
    event_data->thread_event_callback = pthread_self();
    
    volatile int threads_eq = pthread_equal(event_data->thread_event_callback,
                                            event_data->thread_event_data_init);

    if (threads_eq != 0) {
      if (!_message_reported) {
        msg_verbosef("%s is launched on the same thread as the main thread (this is good)\n", __FUNC__);
        _message_reported = true;
      }
    } else {
      exit_msg(stdout,
               ERACE_CONDITION,
               "Race condition detected in %s. "
               "Synchronization primitives will be needed for "
               "nvcd_host_begin() caller's thread wait loop and "
               "the thread for this callback.\n", __FUNC__);
    }
  }

  // actual event handling
  {
    switch (callback_info->callbackSite) {
    case CUPTI_API_ENTER: {
      CUDA_RUNTIME_FN(cudaDeviceSynchronize());

      CUPTI_FN(cuptiSetEventCollectionMode(callback_info->context,
                                           CUPTI_EVENT_COLLECTION_MODE_KERNEL));

      //
      // We try to get all of the event groups we wish to read,
      // but not necessarily at the same time.
      // In this case, it's necessary to repeatedly call the same kernel
      // until
      //           event_data->count_event_groups_read == event_data->num_event_groups
      // is true.
      // The state tracking is handled in this loop,
      // as well as in collect_group_events()
      //

      for (uint32_t i = 0; i < event_data->num_event_groups; ++i) {
        if (event_data->event_group_read_states[i] == CED_EVENT_GROUP_UNREAD) {
          ASSERT(event_data->event_groups[i] != NULL);
          
          CUptiResult err = cuptiEventGroupEnable(event_data->event_groups[i]);

          msg_verbosef("Enabling Group %" PRIu32 " = %p....\n", i, event_data->event_groups[i]);
          
          if (err != CUPTI_SUCCESS) {
            if (err == CUPTI_ERROR_NOT_COMPATIBLE) {
              msg_verbosef("Group %" PRIu32 " out of "
			   "%" PRIu32 " considered not compatible with the current set of enabled groups\n",
			   i,
			   event_data->num_event_groups);

              event_data->event_group_read_states[i] = CED_EVENT_GROUP_DONT_READ;
            } else if (err == CUPTI_ERROR_INVALID_PARAMETER) {
              // This issue (so far) will only occurr if the amount of groups
              // is only one for an event batch. The docs state
              // that this error is thrown when the group passed
              // to cuptiEventGroupEnable() is NULL. So far,
              // this error has only been thrown with non-null
              // group IDs. Still not sure what's going on, here,
              // but obviously the more info the better...
              // At this point, error has only occurred on xsede's pascal 100 node
              // a GTX 960 M. 
              ASSERT(event_data->subscriber != NULL);
              
              msg_warns("BAD_GROUP found");
              event_data->event_group_read_states[i] = CED_EVENT_GROUP_SKIP;
              CUPTI_FN_WARN(err);
            } else {
              CUPTI_FN(err);
            }
          } else {
            event_data->event_groups_enabled[i] = true;
            msg_verbosef("Group %" PRIu32 " enabled.\n", i);
          }
        }
      }

      CUPTI_FN(cuptiDeviceGetTimestamp(callback_info->context,
                                       &event_data->stage_time_nsec_start));
    } break;

    case CUPTI_API_EXIT: {
      size_t finish_time = 0;
      
      CUDA_RUNTIME_FN(cudaDeviceSynchronize());
      CUPTI_FN(cuptiDeviceGetTimestamp(callback_info->context,
                                       &finish_time));

      collect_group_events(event_data);

      MAYBE_GROW_BUFFER_U32_NN(event_data->kernel_times_nsec,
                               event_data->num_kernel_times,
                               event_data->kernel_times_nsec_buffer_length);

      event_data->kernel_times_nsec[event_data->num_kernel_times] =
        finish_time - event_data->stage_time_nsec_start;

      event_data->num_kernel_times++;
    } break;

    default:
      ASSERT(false);
      break;
    }
  }
}

static inline void cupti_event_data_subscribe_callbacks(cupti_event_data_t* e, bool enable) {
  uint32_t u32e = (uint32_t) enable;
  ASSERT(u32e == 0 || u32e == 1);
  
  for (uint32_t i = 0; i < NUM_CUPTI_RUNTIME_CBIDS; ++i) {
    CUPTI_FN(cuptiEnableCallback(u32e,
                                 e->subscriber,
                                 CUPTI_CB_DOMAIN_RUNTIME_API,
                                 g_cupti_runtime_cbids[i]));
  }
}

NVCD_EXPORT void cupti_event_data_subscribe(cupti_event_data_t* e) {
  ASSERT(e != NULL
         /*&& e->subscriber == NULL*/
         && e->initialized);

  if (e->subscriber == NULL) {
    CUPTI_FN(cuptiSubscribe(&e->subscriber,
                            (CUpti_CallbackFunc)cupti_event_callback,
                            (void*) e));
  
  }

  cupti_event_data_subscribe_callbacks(e, true);
}

NVCD_EXPORT void cupti_event_data_unsubscribe(cupti_event_data_t* e) {
  ASSERT(e != NULL && e->initialized && e->subscriber != NULL);

  cupti_event_data_subscribe_callbacks(e, false);

  CUPTI_FN(cuptiUnsubscribe(e->subscriber));
  //
  //  e->subscriber = NULL;
}

static inline void __cupti_event_data_init_base(cupti_event_data_t* e) {
  e->thread_event_data_init = pthread_self();

  e->kernel_times_nsec = zallocNN(sizeof(e->kernel_times_nsec[0]) *
                                  e->kernel_times_nsec_buffer_length);

  init_cupti_event_buffers(e);
}

NVCD_EXPORT void cupti_event_data_init_from_ids(cupti_event_data_t* e,
                                                CUpti_EventID* event_ids,
                                                uint32_t num_event_ids) {
  ASSERT(e != NULL);
  ASSERT(e->cuda_context != NULL);
  ASSERT(e->cuda_device >= 0);
  ASSERT(!e->is_root);

  if (!e->initialized) {
    
    
    // event group initialization
    {
      CUpti_EventGroup eg_buf[MAX_EVENT_GROUPS_PER_EVENT_DATA] = { 0 };
      uint32_t num_egs = 0;
      uint32_t max_egs = MAX_EVENT_GROUPS_PER_EVENT_DATA;
      
      for (uint32_t i = 0; i < num_event_ids; ++i) {
        volatile bool found = find_event_group(e,
                                               &eg_buf[0],
                                               event_ids[i],
                                               max_egs,
                                               &num_egs);

        ASSERT(found);
      }

      ASSERT(num_egs <= max_egs);
      
      fill_event_groups(e,
                        &eg_buf[0],
                        num_egs);
    }

    __cupti_event_data_init_base(e); 
    
    e->initialized = true;
  }
}

NVCD_EXPORT void cupti_event_data_init(cupti_event_data_t* e) {
  ASSERT(e != NULL);
  ASSERT(e->cuda_context != NULL);
  ASSERT(e->cuda_device >= 0);
  ASSERT(e->is_root == true);
  
  if (!e->initialized) {

#ifndef NVCD_OMIT_STANDALONE_EVENT_COUNTER
    init_cupti_event_names(e);
    init_cupti_event_groups(e);

    __cupti_event_data_init_base(e);
#endif

    init_cupti_metric_data(e);
    
    e->initialized = true;
  }
}


NVCD_EXPORT void cupti_event_data_set_null(cupti_event_data_t* e) {
  cupti_event_data_t tmp = CUPTI_EVENT_DATA_INIT;
  memcpy(e, &tmp, sizeof(tmp));
}


// This frees all memory managed by the CUPTI driver
// in addition to our middleware managed memory for the
// given cupti_event_data_t*.
// Note that the only the root cupti_event_data_t* instance
// is supposed to have a non-NULL metric_data pointer,
// so we make an extra check for that and thus handle
// the freeing of the metric_data memory as well.
NVCD_EXPORT void cupti_event_data_free(cupti_event_data_t* e) {
  ASSERT(e != NULL);

  msg_diagf("BEGIN FREE " STRFMT_PTR_VALUE(cupti_event_data_t, e) "\n",
	       e);
  
#if 0
  if (g_nvcd.opt_verbose_output) {
    char* estr = cupti_event_data_to_string(e);    
    ASSERT(estr != NULL);
    msg_diagf("FREEING %s\n", estr);
    free(estr);
  }
#endif

  for (size_t i = 0; i < e->num_event_groups; ++i) {
    msg_diagtab(1); msg_diagf("Checking group " STRFMT_BUFFER_INDEX_SIZE_T(e->event_groups, i) STRFMT_NEWL1, i);
    if (e->event_groups[i] != NULL) {
      msg_diagtab(2); msg_diags("Group is NOT NULL");

      msg_diagtab(2);
      if (e->event_groups_enabled[i] == true) {
        msg_diagtagline(CUPTI_FN(cuptiEventGroupDisable(e->event_groups[i])));
        e->event_groups_enabled[i] = false;
      } else {
	msg_diags("Group already disabled");
      }

      msg_diagtab(2);
      if (cupti_event_group_get_num_events(e->event_groups[i]) > 0) {
        msg_diagtagline(CUPTI_FN(cuptiEventGroupRemoveAllEvents(e->event_groups[i])));
      } else {
	msg_diags("No events in group");
      }
      
      
      msg_diagtab(2); msg_diagtagline(CUPTI_FN(cuptiEventGroupDestroy(e->event_groups[i])));
    } else {
      msg_diagtab(2); msg_diags("Group is NULL");
    }
  }
  
  msg_diagtab(1); msg_diagtagline(safe_free_v(e->event_id_buffer));
  msg_diagtab(1); msg_diagtagline(safe_free_v(e->event_counter_buffer));
  
  msg_diagtab(1); msg_diagtagline(safe_free_v(e->num_events_per_group));
  msg_diagtab(1); msg_diagtagline(safe_free_v(e->num_instances_per_group));
  msg_diagtab(1); msg_diagtagline(safe_free_v(e->num_events_read_per_group));
  
  msg_diagtab(1); msg_diagtagline(safe_free_v(e->event_counter_buffer_offsets));
  msg_diagtab(1); msg_diagtagline(safe_free_v(e->event_id_buffer_offsets));
  msg_diagtab(1); msg_diagtagline(safe_free_v(e->event_group_read_states));
  
  msg_diagtab(1); msg_diagtagline(safe_free_v(e->kernel_times_nsec));
    
  msg_diagtab(1); msg_diagtagline(safe_free_v(e->event_groups));
  
  // TODO: event names may be either a subset of a static buffer
  // initialized in the .data section,
  // or a subset. Should add a flag to determine
  // whether or not the data needs to be freed.  
  if (e->is_root == true) {
    msg_diagtab(1); msg_diags("e->is_root is true");
    if (e->metric_data != NULL) {
      msg_diagtab(2); msg_diags("e->metric_data is NOT NULL");
      ASSERT(e->metric_data->initialized == true);
      
      for (uint32_t i = 0; i < e->metric_data->num_metrics; ++i) {
	msg_diagtab(3);
	msg_diagf("For " STRFMT_UINT32_VALUE(i) ": ", i);
        msg_diagtagline(cupti_event_data_free(&e->metric_data->event_data[i]));
      }

      msg_diagtab(2); msg_diagtagline(safe_free_v(e->metric_data->metric_ids));
      msg_diagtab(2); msg_diagtagline(safe_free_v(e->metric_data->metric_values));
      msg_diagtab(2); msg_diagtagline(safe_free_v(e->metric_data->computed));
      msg_diagtab(2); msg_diagtagline(safe_free_v(e->metric_data->metric_get_value_results));
    }
  } else {
    msg_diagtab(1); msg_diags("e->is_root is false");
  }

  msg_diagtab(1); msg_diagtagline(cupti_event_data_set_null(e));

  msg_diags("END FREE");
}

NVCD_EXPORT void cupti_event_data_begin(cupti_event_data_t* e) {
  ASSERT(e != NULL);

  cupti_event_data_subscribe(e);
}

NVCD_EXPORT void cupti_event_data_end(cupti_event_data_t* e) {
  ASSERT(e != NULL);
  
  cupti_event_data_unsubscribe(e);
}

typedef char name_str_t[256];

NVCD_EXPORT char* cupti_event_get_name(CUpti_EventID eid) {
  name_str_t name = {0};

  size_t sz = sizeof(name);
  
  CUPTI_FN(cuptiEventGetAttribute(eid, CUPTI_EVENT_ATTR_NAME, &sz, &name[0]));

  return strdup(name);
}

NVCD_EXPORT CUpti_EventID* cupti_metric_get_event_ids(CUpti_MetricID metric, uint32_t* num_events) {
  ASSERT(num_events != NULL);
  
  CUPTI_FN(cuptiMetricGetNumEvents(metric, num_events));

  size_t sz = sizeof(CUpti_EventID) * (*num_events);

  CUpti_EventID* event_ids = zallocNN(sz);

  CUPTI_FN(cuptiMetricEnumEvents(metric, &sz, &event_ids[0]));

  return event_ids;
}

NVCD_EXPORT CUpti_MetricID* cupti_metric_get_ids(CUdevice dev, uint32_t* num_metrics) {
  CUPTI_FN(cuptiDeviceGetNumMetrics(dev, num_metrics));

  size_t array_size_bytes = sizeof(CUpti_MetricID) * (*num_metrics);
  
  CUpti_MetricID* metric_ids = mallocNN(array_size_bytes); 

  CUPTI_FN(cuptiDeviceEnumMetrics(dev,
                                  &array_size_bytes,
                                  &metric_ids[0]));

  return metric_ids;
}

NVCD_EXPORT char* cupti_metric_get_name(CUpti_MetricID metric) {
  name_str_t name = {0};

  size_t sz = sizeof(name);
          
  CUPTI_FN(cuptiMetricGetAttribute(metric,
                                   CUPTI_METRIC_ATTR_NAME,
                                   &sz,
                                   (void*) &name[0]));

  return strdup(name);
}

NVCD_EXPORT uint32_t cupti_event_group_get_num_events(CUpti_EventGroup group) {
  ASSERT(group != NULL);

  uint32_t count = 0;
  size_t sz = sizeof(count);

  CUPTI_FN(cuptiEventGroupGetAttribute(group,
                                       CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                       &sz,
                                       (void*) &count));

  return count;
}

NVCD_EXPORT char* cupti_event_data_to_string(cupti_event_data_t* e) {
#define __CED_STR_LEN__ (1 << 14)
  char buffer[__CED_STR_LEN__] = { 0 };

  sprintf(&buffer[0],
          STRFMT_STRUCT_PTR_BEGIN(cupti_event_data_t*, e) STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_PTR_VALUE(CUpti_EventID*, e->event_id_buffer) STRFMT_MEMBER_SEP STRFMT_NEWL1 
          STRFMT_TAB1 STRFMT_PTR_VALUE(uint64_t*, e->event_counter_buffer) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_PTR_VALUE(uint32_t*, e->num_events_per_group) STRFMT_MEMBER_SEP STRFMT_NEWL1
          STRFMT_TAB1 STRFMT_PTR_VALUE(uint32_t*, e->num_events_read_per_group) STRFMT_MEMBER_SEP STRFMT_NEWL1
          STRFMT_TAB1 STRFMT_PTR_VALUE(uint32_t*, e->num_instances_per_group) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_PTR_VALUE(uint32_t*, e->event_counter_buffer_offsets) STRFMT_MEMBER_SEP STRFMT_NEWL1
          STRFMT_TAB1 STRFMT_PTR_VALUE(uint32_t*, e->event_id_buffer_offsets) STRFMT_MEMBER_SEP STRFMT_NEWL1
          STRFMT_TAB1 STRFMT_PTR_VALUE(uint8_t*, e->event_group_read_states) STRFMT_MEMBER_SEP STRFMT_NEWL1
          STRFMT_TAB1 STRFMT_PTR_VALUE(uint8_t*, e->event_groups_enabled) STRFMT_MEMBER_SEP STRFMT_NEWL1         

          STRFMT_TAB1 STRFMT_PTR_VALUE(uint64_t*, e->kernel_times_nsec) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_PTR_VALUE(CUpti_EventGroup*, e->event_groups) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_PTR_VALUE(char * const *, e->event_names) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_PTR_VALUE(cupti_metric_data_t*, e->metric_data) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_UINT64_VALUE(e->stage_time_nsec_start) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_PTR_VALUE(CUcontext, e->cuda_context) STRFMT_MEMBER_SEP STRFMT_NEWL1
          STRFMT_TAB1 STRFMT_INT_VALUE(CUdevice, e->cuda_device, PRId32) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_PTR_VALUE(CUpti_SubscriberHandle, e->subscriber) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_INT_VALUE(pthread_t, e->thread_event_data_init, PRIx32) STRFMT_MEMBER_SEP STRFMT_NEWL1
          STRFMT_TAB1 STRFMT_INT_VALUE(pthread_t, e->thread_event_callback, PRIx32) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_UINT32_VALUE(e->num_event_groups) STRFMT_MEMBER_SEP STRFMT_NEWL1
          STRFMT_TAB1 STRFMT_UINT32_VALUE(e->num_kernel_times) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_UINT32_VALUE(e->count_event_groups_read) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_UINT32_VALUE(e->event_counter_buffer_length) STRFMT_MEMBER_SEP STRFMT_NEWL1
          STRFMT_TAB1 STRFMT_UINT32_VALUE(e->event_id_buffer_length) STRFMT_MEMBER_SEP STRFMT_NEWL1
          STRFMT_TAB1 STRFMT_UINT32_VALUE(e->kernel_times_nsec_buffer_length) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_UINT32_VALUE(e->event_names_buffer_length) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_TAB1 STRFMT_BOOL_STR_VALUE(e->initialized) STRFMT_MEMBER_SEP STRFMT_NEWL1
          STRFMT_TAB1 STRFMT_BOOL_STR_VALUE(e->is_root) STRFMT_MEMBER_SEP STRFMT_NEWL1

          STRFMT_STRUCT_PTR_END(e),
          (void*) e,

          (void*) e->event_id_buffer,
          (void*) e->event_counter_buffer,

          (void*) e->num_events_per_group,
          (void*) e->num_events_read_per_group,
          (void*) e->num_instances_per_group,
          
          (void*) e->event_counter_buffer_offsets,
          (void*) e->event_id_buffer_offsets,
          (void*) e->event_group_read_states,

          (void*) e->event_groups_enabled,
          
          (void*) e->kernel_times_nsec,

          (void*) e->event_groups,
          
          (void*) e->event_names,

          (void*) e->metric_data,

          e->stage_time_nsec_start,

          (void*) e->cuda_context,
          e->cuda_device,

          (uint32_t*) e->subscriber,

          (uint32_t) e->thread_event_data_init,
          (uint32_t) e->thread_event_callback,

          e->num_event_groups,
          e->num_kernel_times,

          e->count_event_groups_read,

          e->event_counter_buffer_length,
          e->event_id_buffer_length,
          e->kernel_times_nsec_buffer_length,

          e->event_names_buffer_length,

          STRVAL_BOOL_STR_VALUE(e->initialized),
          STRVAL_BOOL_STR_VALUE(e->is_root));
  
#undef __CED_STR_LEN__

  return strdup(buffer);
}

NVCD_EXPORT void cupti_event_data_calc_metrics(cupti_event_data_t* e) {
  ASSERT(e != NULL);
  ASSERT(e->is_root == true);
  ASSERT(e->metric_data != NULL);
  
  calc_cupti_metrics(e->metric_data);
}

NVCD_EXPORT bool cupti_event_data_callback_finished(cupti_event_data_t* e) {
  ASSERT(e->count_event_groups_read
         <= e->num_event_groups /* serious problem if this fails */);
  
  return e->count_event_groups_read
    == e->num_event_groups;
}

void cupti_event_data_enum_event_counters(cupti_event_data_t* e,
					  cupti_event_data_enum_event_counters_fn_t fn) {
  ASSERT(e->count_event_groups_read == e->num_event_groups);
  bool keep_iterating = true;
  uint64_t* pcounters = &e->event_counter_buffer[0];
  //g = group
  //e = event
  //j = event count instance
  //ib = id buffer
  //cb = counter buffer
  //nepg = number of events per group
  //nipg = number of instances per group (for each event)  
  uint32_t group = 0;
  while (group < e->num_event_groups && keep_iterating) {
    uint32_t ib_offset = e->event_id_buffer_offsets[group];
    uint32_t cb_offset = e->event_counter_buffer_offsets[group];
  
    uint32_t nepg = e->num_events_per_group[group];
    uint32_t nipg = e->num_instances_per_group[group];

    // if asserts are enabled, then what's listed here is just
    // a direct paste. Scroll further downward for the actual
    // iteration.
    IF_ASSERTS_ENABLED(
	volatile uint32_t next_cb_offset = 0;
	volatile uint32_t next_ib_offset = 0;
	{
	// bounds check ordering for
	// event_counter_buffer_offsets      
	volatile uint32_t prev_cb_offset = (group > 0) ?
      
	  e->event_counter_buffer_offsets[group - 1] :
	  0;

	volatile uint32_t prev_cb_offset_add = (group > 0) ?

	  (e->num_events_per_group[group - 1] *
	   e->num_instances_per_group[group - 1]) :
	  0;

	ASSERT(prev_cb_offset + prev_cb_offset_add == cb_offset);
      }

      {    
	// bounds check ordering for
	// event_id_buffer_offsets
	volatile uint32_t prev_ib_offset = (group > 0) ?

	  e->event_id_buffer_offsets[group - 1] :
	  0;

	volatile uint32_t prev_ib_offset_add = (group > 0) ?
      
	  e->num_events_per_group[group - 1] :
	  0;

	ASSERT(prev_ib_offset + prev_ib_offset_add == ib_offset);
      }

      {
	// used for iterative bounds checking
	next_cb_offset =
	  group < (e->num_event_groups - 1) ?
	  e->event_counter_buffer_offsets[group + 1] :
	  e->event_counter_buffer_length;
      }
  
      {
	// used for iterative bounds checking
	next_ib_offset =
	  group < (e->num_event_groups - 1) ?
	  e->event_id_buffer_offsets[group + 1] :
	  e->event_id_buffer_length;
      });    
    //
    // This is where the rest of the iteration is actually performed.
    //
    uint32_t event = 0;
    while (event < nepg && keep_iterating) {
      ASSERT(ib_offset + event < next_ib_offset);      

      uint32_t event_instance = 0;
      while (event_instance < nipg && keep_iterating) {
	uint32_t k = cb_offset + event_instance * nepg + event;

	ASSERT(k < next_cb_offset);
      
	cupti_enum_event_counter_iteration_t it =
	  {
	   .instance = event_instance,
	   .num_instances = nipg,
	   .value = pcounters[k],
	   .event = e->event_id_buffer[ib_offset + event],
	   .group = e->event_groups[group]
	  };

	keep_iterating = fn(&it);
	event_instance++;
      }
      event++;
    }
    group++;
  }
}
