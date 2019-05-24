#include "nvci.h"
#include <stdio.h>

#include "commondef.h"
#include "gpu.h"
#include "cupti_lookup.h"
#include "list.h"
#include "env_var.h"

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

typedef struct nvcd {
	CUdevice* devices;
	CUcontext* contexts;
	int num_devices;
} nvcd_t;

static void nvcd_init_cuda(nvcd_t* nvcd) {
	CUDA_DRIVER_FN(cuInit(0));
	
	CUDA_RUNTIME_FN(cudaGetDeviceCount(&nvcd->num_devices));

	nvcd->devices =
		zallocNN(sizeof(*(nvcd->devices)) * nvcd->num_devices);

	nvcd->contexts =
		zallocNN(sizeof(*(nvcd->contexts)) * nvcd->num_devices);
	
	for (int i = 0; i < nvcd->num_devices; ++i) {
		CUDA_DRIVER_FN(cuDeviceGet(&nvcd->devices[i], i));
		CUDA_DRIVER_FN(cuCtxCreate(&nvcd->contexts[i], 0, nvcd->devices[i]));
	}
}

//
// Cupti Event
//

static cupti_event_data_t g_event_data = {
	.event_id_buffer = NULL, 
	.event_counter_buffer = NULL, 

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

	.cuda_context = NULL,
	.cuda_device = CU_DEVICE_INVALID,

	.subscriber = NULL,
	
	.num_event_groups = 0,
	.num_kernel_times = 0,

	.count_event_groups_read = 0,
	
	.event_counter_buffer_length = 0,
	.event_id_buffer_length = 0,
	.kernel_times_nsec_buffer_length = 10, // default; will increase as necessary at runtime

	.event_names_buffer_length = 0,

	.initialized = false
};

static nvcd_t g_nvcd = {
	.devices = NULL,
	.contexts = NULL,
	.num_devices = 0
};

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

void cleanup() {
	cupti_event_data_free(&g_event_data);
	
	cupti_name_map_free();

	for (int i = 0; i < g_nvcd.num_devices; ++i) {
		ASSERT(g_nvcd.contexts[i] != NULL);
		CUDA_DRIVER_FN(cuCtxDestroy(g_nvcd.contexts[i]));
	}
}

NVCD_EXPORT void nvcd_init() {
	nvcd_init_cuda(&g_nvcd);
}

NVCD_EXPORT void nvcd_host_begin() {	
	g_event_data.cuda_context = g_nvcd.contexts[0];
	g_event_data.cuda_device = g_nvcd.devices[0];

	g_event_data.thread_host_begin = pthread_self();
	
	cupti_event_data_init(&g_event_data);
	cupti_event_data_begin(&g_event_data);
}

NVCD_EXPORT bool nvcd_host_finished() {
	
	ASSERT(g_event_data.count_event_groups_read
				 <= g_event_data.num_event_groups /* serious problem if this fails */);
	
	return g_event_data.count_event_groups_read
		== g_event_data.num_event_groups; 
}

NVCD_EXPORT void nvcd_host_finalize() {
	cupti_event_data_end(&g_event_data);
}

NVCD_EXPORT void nvcd_terminate() {
	cleanup();
}
