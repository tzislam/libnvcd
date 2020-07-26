#include "gpu.h"

//#define NVCD_HEADER_IMPL
//#include <nvcd/nvcd.cuh>
//#undef NVCD_HEADER_IMPL

extern "C" {

__global__ void nvcd_kernel_test() {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;

  int num_threads = blockDim.x * gridDim.x;

  if (thread == 0) {
    
  }

  if (thread < num_threads) {
    //   nvcd_device_begin(thread);

    volatile int number = 0;

    for (int i = 0; i < 100000; ++i) {
      number += i;
    }

    //    nvcd_device_end(thread);
  }
}
  
__host__ void gpu_call() {
    int num_threads = 1024;
//    nvcd_host_begin(num_threads);
    
    int nblock = 4;
    int threads = num_threads / nblock;
    nvcd_kernel_test<<<nblock, threads>>>();
    

//    const char* name = *(const char**)((uintptr_t)(nvcd_kernel_test) + 8);

//    printf("the kernel name = %s\n", name);
    
//    nvcd_run(nvcd_kernel_test, nblock, threads);

//    nvcd_host_end();

    num_threads = 2048;
//    nvcd_host_begin(num_threads);

    threads = num_threads / nblock;
    nvcd_kernel_test<<<nblock, threads>>>();

//    nvcd_run(nvcd_kernel_test, nblock, threads);

//    nvcd_host_end();
}

}
