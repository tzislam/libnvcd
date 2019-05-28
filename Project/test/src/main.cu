#include <device.cuh>

#include "gpu.h"

int main() {
  nvcd_init();

  int threads = 16;

  gpu_test_matrix_vec_mul(threads);

  nvcd_report();

  nvcd_terminate();

  return 0;
}

