#include "nvcd.h"
#include <stdio.h>

#include "commondef.h"
#include "cupti_lookup.h"
#include "list.h"
#include "env_var.h"

#include "device.cuh"

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

//
// nvcd base data
//

typedef struct nvcd {
  CUdevice* devices;
  CUcontext* contexts;

  int num_devices;
  
  bool32_t initialized;
} nvcd_t;

static void nvcd_init_cuda(nvcd_t* nvcd) {
  if (!nvcd->initialized) {
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

    nvcd->initialized = true;
  }
}

static nvcd_t g_nvcd = {
  .devices = NULL,
  .contexts = NULL,

  .num_devices = 0,

  .initialized = false
};

//
// kernel thread
// 

typedef struct kernel_thread_data {
  clock64_t* times;

  uint32_t* smids;
  
  size_t num_cuda_threads;

  bool32_t initialized;
  
} kernel_thread_data_t;

static void kernel_thread_data_init(kernel_thread_data_t* k, int num_cuda_threads) {
  ASSERT(k != NULL);

  if (!k->initialized) {
    k->num_cuda_threads = (size_t)num_cuda_threads;

    k->times = zallocNN(sizeof(k->times[0]) * k->num_cuda_threads);
    k->smids = zallocNN(sizeof(k->smids[0]) * k->num_cuda_threads);

    k->initialized = true;
  }
}

static void kernel_thread_data_free(kernel_thread_data_t* k) {
  ASSERT(k != NULL);

  safe_free_v(k->times);
  safe_free_v(k->smids);

  k->initialized = false;
}

static void kernel_thread_data_report(kernel_thread_data_t* k) {
  ASSERT(k != NULL);
  
  puts("============PER-THREAD KERNEL DATA============");

  for (size_t i = 0; i < k->num_cuda_threads; ++i) {
    printf("---\nThread %" PRIu64 ":\n"
           "\tTime\t= %" PRId64 " nanoseconds\n"
           "\tSM ID\t= %" PRIu32 "\n",
           i,
           k->times[i],
           k->smids[i]);
  }
  puts("==============================================");
}

static kernel_thread_data_t g_kernel_thread_data = {
  .times = NULL,
  .smids = NULL,
  
  .num_cuda_threads = 0,

  .initialized = false
};

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

NVCD_EXPORT void nvcd_report() {
  ASSERT(g_event_data.initialized == true);
  
  cupti_report_event_data(&g_event_data);
}

NVCD_EXPORT void nvcd_init() {
  nvcd_init_cuda(&g_nvcd);
}

NVCD_EXPORT void nvcd_host_begin(int num_cuda_threads) {  
  ASSERT(g_nvcd.initialized == true);

  nvcd_device_init_mem(num_cuda_threads);

  kernel_thread_data_init(&g_kernel_thread_data, num_cuda_threads);
  
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

NVCD_EXPORT void nvcd_host_end() {
  ASSERT(g_kernel_thread_data.initialized == true);
  ASSERT(g_nvcd.initialized == true);
  
  nvcd_device_get_ttime(g_kernel_thread_data.times);
  nvcd_device_get_smids(g_kernel_thread_data.smids);

  kernel_thread_data_report(&g_kernel_thread_data);
  kernel_thread_data_free(&g_kernel_thread_data);
  
  cupti_event_data_end(&g_event_data);
  nvcd_device_free_mem();
}

NVCD_EXPORT void nvcd_terminate() {
  cleanup();
}
