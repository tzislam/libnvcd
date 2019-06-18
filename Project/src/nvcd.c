#include "nvcd/nvcd.h"
#include "nvcd/cupti_lookup.h"

static cupti_event_data_t g_event_data = CUPTI_EVENT_DATA_NULL;

void nvcd_init_events(CUdevice device, CUcontext context) {
  g_event_data.cuda_context = context;
  g_event_data.cuda_device = device;
  g_event_data.is_root = true;
    
  cupti_event_data_init(&g_event_data);
}

void nvcd_calc_metrics() {
  cupti_event_data_calc_metrics(&g_event_data);
}

void nvcd_free_events() {
  cupti_event_data_free(&g_event_data);
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

#endif
