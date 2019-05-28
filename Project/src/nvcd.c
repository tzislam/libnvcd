#include "nvcd.h"
#include <stdio.h>

#include "commondef.h"
#include "cupti_lookup.h"
#include "list.h"
#include "env_var.h"

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
