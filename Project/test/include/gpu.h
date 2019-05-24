#ifndef __GPU_H__
#define __GPU_H__

#include <commondef.h>
#include <util.h>

#include <stdio.h>

C_LINKAGE_START



__global__ void gpu_kernel();

__host__ void gpu_test();


__global__ void gpu_kernel_matrix_vec_mul_int(int n,
                                              int m,
                                              int* q,
                                              int* u,
                                              int* v,
                                              clock64_t* d_times);

__host__ void gpu_test_matrix_vec_mul(int num_threads, clock64_t* h_exec_times);

C_LINKAGE_END

#endif
