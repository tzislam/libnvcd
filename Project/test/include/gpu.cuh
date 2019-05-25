#ifndef __GPU_CUH__
#define __GPU_CUH__

#include <commondef.h>
#include <stdio.h>

C_LINKAGE_START

typedef long long int clock64_t;

#define GPU_ASSERT(condition_expr) assert_cond_impl(condition_expr, #condition_expr, __LINE__) 

__device__ bool assert_cond_impl(bool condition, const char* message, int line);

/*
 * transforms an M x 1 vector u 
 * by an N x M matrix q
 * resulting in an N x 1 vector v
 *
 * qu = v 
 */

__global__ void gpu_kernel_matrix_vec_mul_int(int n,
                                              int m,
                                              int* q,
                                              int* u,
                                              int* v,
                                              clock64_t* d_times);


C_LINKAGE_END

#endif