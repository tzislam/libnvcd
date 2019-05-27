#include <nvcd.h>

#include "gpu.h"

int main() {
  nvcd_init();

  int threads = 1024;

  gpu_test_matrix_vec_mul(threads);

  nvcd_report();

  nvcd_terminate();

  return 0;
}

