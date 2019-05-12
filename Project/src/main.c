#include "commondef.h"
#include "gpu.h"

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
 * CUPTI events
 */


typedef struct cupti_event_data {
	CUpti_EventID* ids;
	CUpti_EventGroup group;
	uint8_t num_events;
	uint8_t initialized;
} cupti_event_data_t;


static CUpti_EventID g_event_id_backing[NUM_CUPTI_EVENTS];

static cupti_event_data_t g_default_event_data = {
	&g_event_id_backing[0],
	0,
	NUM_CUPTI_EVENTS,
	false
};



cupti_event_data_t* default_event_data() {
	if (!g_default_event_data.initialized) {
		// create group, assign to g_default_event_data
		// find ids for all event counters
	}

	return &g_default_event_data;
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

struct cupti_data {
	
	
};

struct profile_data {
	struct cupti_data cupti;
	
	string_micro_t* device_names;
	CUdevice* device_ids; 
	
	profile_time_t start;
	profile_time_t time; /* TODO: add darray since time slices will need to be taken */

	int num_devices;
	int device; /* Defaults to zero */
	
	uint8_t needs_init;
};

static struct profile_data* g_data = NULL;

void string_list_free(char** list, size_t sz) {
	ASSERT(list != NULL);
	
	for (size_t i = 0; i < sz; ++i) {
	  ASSERT(list[i] != NULL);

		free(list[i]);
		list[i] = NULL;
	}

	free(list);
	list = NULL;
}

struct profile_data* default_profile_data() {
	if (g_data == NULL) {
		{
			g_data = zalloc(sizeof(*g_data));
			ASSERT(g_data != NULL);

			g_data->needs_init = true;
		}
	}

	return g_data;
}

void profile_data_print(struct profile_data* data) {

	
	for (int i = 0; i < data->num_devices; ++i) {
		printf("device: %i. device id: 0x%x. name: \"%s\"\n",
					 i,
					 data->device_ids[i],
					 data->device_names[i]);
	}
}
 

void cupti_benchmark_start() {
	struct profile_data* data = default_profile_data();

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
		}

		// INIT CUPTI HERE
		
		profile_data_print(data);
		
		data->needs_init = false;
	}

	data->start = profile_time();
}

void cupti_benchmark_end() {
	struct profile_data* data = default_profile_data();
	
	data->time = profile_time() - data->start;

	printf("time taken: %" PTIME_FMT ".\n", data->time);

	
}

int main() {
	if (g_test_params.run) {
		test_env_parse();
	}

	cupti_benchmark_start();
	
	gpu_test_matrix_vec_mul();
	
	gpu_test();
	
	CUDA_RUNTIME_FN(cudaDeviceSynchronize());
	
	return 0;
}
