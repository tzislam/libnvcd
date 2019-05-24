#include <nvcd.h>

#include "gpu.h"

int main() {
  nvcd_init();
  
  int threads = 1024;
  
  clock64_t* thread_times = (clock64_t*) zallocNN(sizeof(thread_times[0]) * threads);

  NVCD_EXEC_KERNEL(gpu_test_matrix_vec_mul(threads, &thread_times[0]));
  
  for (int i = 0; i < threads; ++i) {
    printf("[%i] time: %llu\n", i, thread_times[i]);
  }

  nvcd_report();
  
  nvcd_terminate();
  
  return 0;
}

