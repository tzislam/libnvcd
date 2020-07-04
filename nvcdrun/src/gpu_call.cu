#include "gpu.h"

#define NVCD_HEADER_IMPL
#include <nvcd/nvcd.cuh>
#undef NVCD_HEADER_IMPL

extern "C" {

__host__ void gpu_call() {
    int num_threads = 1024;
    nvcd_host_begin(num_threads);
    
    int nblock = 4;
    int threads = num_threads / nblock;    

    NVCD_KERNEL_EXEC_KPARAMS_2(nvcd_kernel_test, nblock, threads);

    nvcd_host_end();

    num_threads = 2048;
    nvcd_host_begin(num_threads);

    threads = num_threads / nblock;

    NVCD_KERNEL_EXEC_KPARAMS_2(nvcd_kernel_test, nblock, threads);

    nvcd_host_end();
}

}