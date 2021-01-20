#include "gpu.h"

#include <libnvcd.h>
#include <stdio.h>

LIBNVCD_STORE_FUNCTION_POINTERS_HERE;

extern "C" {  
  __global__ void kernel2() {
    volatile int i = 200000;
    while (i > 0) {
      i--;
    }
  }

  __global__ void kernel3() {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    if (thread < num_threads) {
      volatile unsigned i = 0;
      while (i < 100000) {
	i++;
      }
    }
  }
  
  __global__ void nvcd_kernel_test() {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;

    int num_threads = blockDim.x * gridDim.x;

    if (thread == 0) {
    
    }

    if (thread < num_threads) {
      volatile int number = 0;

      for (int i = 0; i < 100000; ++i) {
	number += i;
      }
    }
  }
  
  __host__ void gpu_call(unsigned timeflags, unsigned repeat) {

    libnvcd_load();

    puts("=======================================================================");
    printf("[nvcdrun] running test kernels within two separate regions. timeflags = %s\n",
	   libnvcd_time_str(timeflags));
    puts("=======================================================================");
    
    libnvcd_time(timeflags);
    libnvcd_begin("REGION A");    
    
    int num_threads = 1024;
    
    int nblock = 4;
    int threads = num_threads / nblock;

    for (unsigned i = 0; i < repeat; ++i) {
      nvcd_kernel_test<<<nblock, threads>>>();   

      //      num_threads = 2048;

      //      threads = num_threads / nblock;
      //   nvcd_kernel_test<<<nblock, threads>>>();

      kernel3<<<nblock, threads>>>();
    }
    
    libnvcd_end();

    libnvcd_begin("REGION B");
    
    num_threads = 1024;
    
    nblock = 4;
    threads = num_threads / nblock;

    for (unsigned i = 0; i < repeat; ++i) {
      nvcd_kernel_test<<<nblock, threads>>>();   

      num_threads = 2048;

      threads = num_threads / nblock;
      nvcd_kernel_test<<<nblock, threads>>>();

      kernel2<<<nblock, threads>>>();
    }
    
    libnvcd_end();

    puts("[nvcdrun] now for the final kernel run, outside of the test regions");
    
    threads = num_threads / nblock;
    nvcd_kernel_test<<<nblock, threads>>>();

    libnvcd_time_report();
  }
}
