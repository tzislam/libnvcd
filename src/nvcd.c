#include "nvcd/nvcd.h"
#include "nvcd/cupti_util.h"
#include "nvcd/util.h"

static cupti_event_data_t g_event_data = CUPTI_EVENT_DATA_INIT;

nvcd_t g_nvcd =
  {
   .devices = NULL,
   .contexts = NULL,
   .contexts_ext = NULL,
   .device_uuids = NULL,
   .num_devices = 0,
   .initialized = false,
   .opt_verbose_output = false,
   .opt_diagnostic_output = false
  };

static inline void print_device_info(int device_index) {
  msg_userf("GPU %i\n", device_index);

  uint8_t* uuid = (uint8_t*) &g_nvcd.device_uuids[device_index].bytes[0];

  msg_userf("\tgpu_name = %s\n", &g_nvcd.device_names[device_index][0]);

  msg_userf("\tgpu_uuid = GPU-%x%x%x%x-%x%x-%x%x-%x%x-%x%x%x%x%x%x\n",
	    uuid[0],uuid[1],uuid[2],uuid[3],
	    uuid[4],uuid[5],uuid[6],uuid[7],
	    uuid[8],uuid[9],uuid[10],uuid[11],
	    uuid[12],uuid[13],uuid[14],uuid[15]);
}

void nvcd_init_cuda() {
  if (!g_nvcd.initialized) {
    CUDA_DRIVER_FN(cuInit(0));
  
    CUDA_RUNTIME_FN(cudaGetDeviceCount(&g_nvcd.num_devices));

    g_nvcd.devices = (CUdevice*)zallocNN(sizeof(*(g_nvcd.devices)) *
					g_nvcd.num_devices);

    g_nvcd.contexts = (CUcontext*)zallocNN(sizeof(*(g_nvcd.contexts)) *
					  g_nvcd.num_devices);

    g_nvcd.contexts_ext = (bool32_t*)zallocNN(sizeof(g_nvcd.contexts_ext[0]) *
					      g_nvcd.num_devices);

    g_nvcd.device_names = (char**)zallocNN(sizeof(*(g_nvcd.device_names)) *
					  g_nvcd.num_devices);

    g_nvcd.device_uuids = (uuid_t*)zallocNN(sizeof(g_nvcd.device_uuids[0]) *
					   g_nvcd.num_devices);

    const size_t MAX_STRING_LENGTH = 128;
      
    for (int i = 0; i < g_nvcd.num_devices; ++i) {
      CUDA_DRIVER_FN(cuDeviceGet(&g_nvcd.devices[i], i));

      //
      // NOTE: this obviously isn't device specific.
      // Should look into cuDevicePrimaryCtxRetain/cuDevicePrimaryCtxRelease
      // for that. It may be that we can get away with those alone and nothing else,
      // but I'm not sure. Will need time to read more on context management...
      //         
      CUDA_DRIVER_FN(cuCtxGetCurrent(&g_nvcd.contexts[i]));

      g_nvcd.contexts_ext[i] = g_nvcd.contexts[i] != NULL;
      
      if (g_nvcd.contexts_ext[i] == false) {
	CUDA_DRIVER_FN(cuCtxCreate(&g_nvcd.contexts[i],
				   0,
				   g_nvcd.devices[i]));
      }

      ASSERT(g_nvcd.contexts[i] != NULL);
        
      g_nvcd.device_names[i] = (char*) zallocNN(sizeof(g_nvcd.device_names[i][0]) *
					       MAX_STRING_LENGTH);
        
      CUDA_DRIVER_FN(cuDeviceGetName(&g_nvcd.device_names[i][0],
				     MAX_STRING_LENGTH,
				     g_nvcd.devices[i]));

      CUDA_DRIVER_FN(cuDeviceGetUuid(&g_nvcd.device_uuids[i],
				     g_nvcd.devices[i]));

      print_device_info(i);
    }
    
    g_nvcd.initialized = true;
  }
}

void nvcd_init_events(CUdevice device, CUcontext context) {
  g_event_data.cuda_context = context;
  g_event_data.cuda_device = device;
  g_event_data.is_root = true;

  cupti_event_data_init(&g_event_data);
}

void nvcd_calc_metrics() {
  cupti_event_data_calc_metrics(&g_event_data);
}

// We delay freeing g_event_data's memory from the most recent call to nvcd_init_events, since it's stored
// by the client and thus can be read at the user's request.
// However, we still need ensure proper assumptions are met by the cupti module,
// which expects a NULL cupti_event_data_t on init.
void nvcd_reset_event_data() {
  cupti_event_data_free(&g_event_data);
  cupti_event_data_set_null(&g_event_data);
}

cupti_event_data_t* nvcd_get_events() {
  return &g_event_data;
}

//#include "device.cuh"


//
// nvcd base data
//


//
// kernel thread
// 


//
// Cupti Event
//



#if 0
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
    msg_verbosef("Testing %s. Expecting %s with a count of %lu\n",
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
      msg_verbosef("env_var_list_read for %s returned NULL\n", str);
    }
  } else {
    ASSERT(count == expected_count);
    ASSERT(list != NULL);

    for (size_t i = 0; i < count; ++i) {
      msg_verbosef("[%lu]: %s\n", i, list[i]);
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

#endif
