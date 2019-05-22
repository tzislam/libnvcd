#include "commondef.h"
#include "gpu.h"
#include "cupti_lookup.h"
#include "list.h"

#include <ctype.h>
#include <cupti.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#ifndef ENV_PREFIX
#define ENV_PREFIX "BENCH_"
#endif

#define ENV_DELIM ':'

/*
 * cupti event
 */ 

static CUpti_runtime_api_trace_cbid g_cupti_runtime_cbids[] = {
	CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020,
	CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000
};

#define NUM_CUPTI_RUNTIME_CBIDS (sizeof(g_cupti_runtime_cbids) / sizeof(g_cupti_runtime_cbids[0]))

static cupti_event_data_t g_cupti_events_2x = {
	.event_id_buffer = NULL, // event_id_buffer
	NULL, // event_counter_buffer
	NULL, // num_events_per_group
	NULL, // num_instances_per_group
	NULL, // event_counter_buffer_offsets
	NULL, // event_id_buffer_offsets
	NULL, // kernel_times_nsec_start
	NULL, // kernel_times_nsec_end
	NULL, // event_groups
	&g_cupti_event_names_2x[0], // event_names
	0, // stage_time_nsec_start
	0, // stage_time_nsec_end
	NULL, // context
	NUM_CUPTI_EVENTS_2X, // num_events
	0, // num_event_groups
	0, // num_kernel_times
	0, // event_counter_buffer_length
	0, // event_id_buffer_length
	0 // kernel_times_nsec_buffer_length
};

static CUpti_SubscriberHandle g_cupti_subscriber = NULL;

void collect_group_events(cupti_event_data_t* e) {
	for (uint32_t i = 0; i < e->num_event_groups; ++i) {

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
	}
}

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

	{
		cupti_event_data_t* event_data = (cupti_event_data_t*) userdata;
		
		switch (callback_info->callbackSite) {
		case CUPTI_API_ENTER: {
			CUDA_RUNTIME_FN(cudaDeviceSynchronize());

			CUPTI_FN(cuptiSetEventCollectionMode(callback_info->context,
																					 CUPTI_EVENT_COLLECTION_MODE_KERNEL));
			
			for (size_t i = 0; i < event_data->num_event_groups; ++i) {
				CUPTI_FN(cuptiEventGroupEnable(event_data->event_groups[i]));
			}

			CUPTI_FN(cuptiDeviceGetTimestamp(event_data->context,
																			 &event_data->stage_time_nsec_start));
		} break;

		case CUPTI_API_EXIT: {
			CUDA_RUNTIME_FN(cudaDeviceSynchronize());

			CUPTI_FN(cuptiDeviceGetTimestamp(event_data->context,
																			 &event_data->stage_time_nsec_end));
			
			uint32_t num_instances = 0;
			size_t value_size = sizeof(num_instances);
			CUPTI_FN(cuptiEventGroupGetAttribute(event_data->context,
																					 CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
																					 &value_size, &num_instances));
			
			
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
													&g_cupti_events_2x));

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
	
	for (uint32_t i = 0; i < e->num_events; ++i) {
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

	// fill our event groups buffer
	{
		e->num_event_groups = num_egs;
		e->event_groups = zallocNN(sizeof(e->event_groups[0]) * e->num_event_groups);

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
	safe_free_v(e->event_counter_buffer_offsets);
	safe_free_v(e->event_id_buffer_offsets);

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
	free_cupti_event_data(&g_cupti_events_2x);
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
	struct env_var_list_scan_ctx ctx = {0};
	
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
 

void cupti_benchmark_start() {
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

		init_cupti_event_data(cuda_get_context(), cuda_get_device(), &g_cupti_events_2x);

		cupti_subscribe();
		
		profile_data_print(data);
		
		data->needs_init = false;
	}

	data->start = profile_time();
}

void cupti_benchmark_end() {
	profile_data_t* data = default_profile_data();
	
	data->time = profile_time() - data->start;

	printf("time taken: %" PTIME_FMT ".\n", data->time);	
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

	int threads = 1024;
	
	//	cupti_benchmark_start(threads);

	clock64_t* thread_times = zallocNN(sizeof(thread_times[0]) * threads);
	gpu_test_matrix_vec_mul(threads, thread_times);
	
	//	gpu_test();
	
	CUDA_RUNTIME_FN(cudaDeviceSynchronize());

	for (int i = 0; i < threads; ++i) {
		printf("[%i] time: %llu\n", i, thread_times[i]);
	}
	
	//cupti_benchmark_end();

	//cleanup();
	
	return 0;
}
