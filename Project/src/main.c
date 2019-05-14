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
	NULL,
	NULL,
	NULL,
	&g_cupti_event_names_2x[0],
	0,
	NUM_CUPTI_EVENTS_2X,
	0
};

static CUpti_SubscriberHandle g_cupti_subscriber = NULL;

void collect_group_events(cupti_event_data_t* event_data, uint32_t group_index) {
	uint32_t num_instances = 0;
	uint64_t* values = NULL;
	size_t value_size = sizeof(num_instances);


	CUPTI_FN(cuptiEventGroupGetAttribute(event_data->event_groups[group_index],
																			 CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
																			 &value_size,
																			 &num_instances));

	values = NOT_NULL(zalloc(sizeof(uint64_t) * num_instances));

	
	
	free(values);
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
			// TODO: start global (per-cupti_event_data_t) timer
		} break;

		case CUPTI_API_EXIT: {
			CUDA_RUNTIME_FN(cudaDeviceSynchronize());
			// TODO: stop global (per-cupti_event_data_t) timer here
		} break;

		default:
			ASSERT(false);
			break;
		}
	}
}

void cupti_subscribe() {
	CUPTI_FN(cuptiSubscribe(&g_cupti_subscriber, (CUpti_CallbackFunc)cupti_event_callback, &g_cupti_events_2x));

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

void cupti_eventlist_push(cupti_elist_node_t** root, cupti_index_t event_name_index, CUpti_EventID event_id) {
	ASSERT(root != NULL);

	cupti_elist_node_t* new_node = NOT_NULL(malloc(sizeof(*new_node)));

	new_node->self.next = NULL;
	new_node->event_id = event_id;
	new_node->event_name_index = event_name_index;

	#if 0
	{
		ASSERT((size_t)event_name_index < g_cupti_events_2x.num_events);
		
		printf("Adding Node: %s @ [%i]:0x%x |",
					 g_cupti_events_2x.event_names[new_node->event_name_index],
					 new_node->event_name_index,
					 new_node->event_id);
	}
	#endif
	
	list_push_fn_impl(root, new_node, cupti_elist_node_t, self);
}

void cupti_eventlist_free_node(cupti_elist_node_t* n) {
	n->event_id = V_UNSET;
	n->event_name_index = V_UNSET;
}

void cupti_eventlist_free(cupti_elist_node_t* root) {
	list_free_fn_impl(root, cupti_elist_node_t, cupti_eventlist_free_node, self);
}

void init_cupti_event_data(CUcontext ctx, CUdevice dev, cupti_event_data_t* e, size_t num_threads) {
	ASSERT(ctx != NULL);
	ASSERT(dev >= 0);

	e->num_event_groups = 24; /* statically allocate this for now; can make more sophisticated later */
	e->num_threads = num_threads;
	
	e->counter_buffer = NOT_NULL(zalloc(sizeof(e->counter_buffer[0]) * e->num_events * e->num_threads));

	e->event_groups = NOT_NULL(malloc(sizeof(e->event_groups[0]) * e->num_event_groups));
	e->event_group_id_lists = NOT_NULL(malloc(sizeof(e->event_group_id_lists[0]) * e->num_event_groups));
	
	MEMSET_NULL(e->event_groups, sizeof(e->event_groups[0]) * e->num_event_groups);
	MEMSET_NULL(e->event_group_id_lists, sizeof(e->event_group_id_lists[0]) * e->num_event_groups);
	
	for (size_t i = 0; i < e->num_events; ++i) {
		CUpti_EventID event_id = V_UNSET;
		
		CUptiResult err = cuptiEventGetIdFromName(dev,
																							e->event_names[i],
																							&event_id);

		bool available = true;
			
		if (err != CUPTI_SUCCESS) {
			if (err == CUPTI_ERROR_INVALID_EVENT_NAME) {
				available = false;
			} else {
				/* trigger exit */
				CUPTI_FN(err);
			}
		}

		size_t event_group = 0;
		
		if (available) {
			size_t j = 0;
			err = CUPTI_ERROR_NOT_COMPATIBLE;

			bool iterating = j < e->num_event_groups;
			
			while (iterating) {
				if (e->event_groups[j] == NULL) {
					CUPTI_FN(cuptiEventGroupCreate(ctx, &e->event_groups[j], 0));
				}

				err = cuptiEventGroupAddEvent(e->event_groups[j], event_id);
				
				event_group = j;
				j++;

				if (j == e->num_event_groups
						|| !(err == CUPTI_ERROR_MAX_LIMIT_REACHED
								 || err == CUPTI_ERROR_NOT_COMPATIBLE)) {
					iterating = false;
				}
			}

			ASSERT(j <= e->num_event_groups);
			
			/* trigger exit if we still error out */
			CUPTI_FN(err);
			
			if (j < e->num_event_groups) {
				cupti_eventlist_push(&e->event_group_id_lists[event_group],
														 (cupti_index_t) i,
														 event_id);
			}
		}

		printf("(%s) index %lu, group_index %lu => %s:0x%x\n",
					 available ? "available" : "unavailable",
					 i,
					 event_group,
					 e->event_names[i],
					 event_id);
	}
}

void free_cupti_event_data(cupti_event_data_t* e) {
	ASSERT(e != NULL);
	ASSERT(e->event_groups != NULL);
	
	for (size_t i = 0; i < e->num_event_groups; ++i) { 
		if (e->event_groups[i] != NULL) {
			CUPTI_FN(cuptiEventGroupRemoveAllEvents(e->event_groups[i]));
			CUPTI_FN(cuptiEventGroupDestroy(e->event_groups[i]));

			ASSERT(e->event_group_id_lists[i] != NULL);
			cupti_eventlist_free(e->event_group_id_lists[i]);
		}
	}
	
	free(e->event_groups);

	ASSERT(e->counter_buffer != NULL);
	free(e->counter_buffer);

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
#if 0
 void profile_timelist_push(cupti_elist_node_t** root, profile_time_t start, profile_time_t end) {
	 ASSERT(root != NULL);

	cupti_elist_node_t* new_node = NOT_NULL(malloc(sizeof(*new_node)));

	new_node->self.next = NULL;
	new_node->event_id = event_id;
	new_node->event_name_index = event_name_index;

		
	list_push_fn_impl(root, new_node, cupti_elist_node_t, self);
}

void cupti_eventlist_free_node(cupti_elist_node_t* n) {
	n->event_id = V_UNSET;
	n->event_name_index = V_UNSET;
}

void cupti_eventlist_free(cupti_elist_node_t* root) {
	list_free_fn_impl(root, cupti_elist_node_t, cupti_eventlist_free_node, self);
}
 #endif
 
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
 

void cupti_benchmark_start(size_t num_threads) {
	profile_data_t* data = default_profile_data();

	if (data->needs_init) {
		CUDA_DRIVER_FN(cuInit(0));

		{
			CUDA_RUNTIME_FN(cudaGetDeviceCount(&data->num_devices));

			data->device_ids =
				NOT_NULL(zalloc(sizeof(*(data->device_ids)) * data->num_devices));

			data->device_names =
				NOT_NULL(zalloc(sizeof(*(data->device_names)) * data->num_devices));
		
			for (int i = 0; i < data->num_devices; ++i) {
				CUDA_DRIVER_FN(cuDeviceGet(&data->device_ids[i], i));
			
				CUDA_DRIVER_FN(cuDeviceGetName(data->device_names[i],
																			 sizeof(data->device_names[i]) - 1,
																			 data->device_ids[i]));
			}

			cuda_set_device(data->device_ids[0]);
		}

		init_cupti_event_data(cuda_get_context(), cuda_get_device(), &g_cupti_events_2x, num_threads);

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

	clock64_t* thread_times = NOT_NULL(zalloc(sizeof(thread_times[0]) * threads));
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
