/* This program uses the host CURAND API to generate 100
 * pseudorandom floats.
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand.h>

#include <nvcd/nvcd.cuh>

static void _curand_fn(curandStatus_t x, const char* expr, int line) {
  if (x != CURAND_STATUS_SUCCESS) {
    printf("Error for %s at %i. code received: %u", expr, line, x);
    exit((int) x);
  } 
}

#define CURAND_CALL(x) _curand_fn(x, #x, __LINE__)

template <typename T>
T* dev_gen_random(size_t n, const T rand_min, const T rand_max) {
  if (sizeof(T) != sizeof(unsigned int)) {
    puts("Bad size received: must be 4 bytes in size and NOT floating point\n");
    exit(0);
  }
  
  curandGenerator_t gen;
  
  T *devData, *hostData;
  
  /* Allocate n ints on host */
  hostData = (T *)calloc(n, sizeof(T));

  /* Allocate n ints on device */
  CUDA_RUNTIME_FN(cudaMalloc((void **)&devData, n * sizeof(T)));

  /* Create pseudo-random number generator */
  CURAND_CALL(curandCreateGenerator(&gen,
                                    CURAND_RNG_PSEUDO_DEFAULT));

  /* Set seed */
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

  /* Generate n ints on device */
  CURAND_CALL(curandGenerate(gen, devData, n));

  /* Copy device memory to host */
  CUDA_RUNTIME_FN(cudaMemcpy(hostData,
                       devData,
                       n * sizeof(T),
                       cudaMemcpyDeviceToHost));

  /* Set seed */
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long) time(NULL)));

  /* Generate n Ts on device */
  CURAND_CALL(curandGenerate(gen,
                             devData,
                             n));

  /* Copy device memory to host */
  CUDA_RUNTIME_FN(cudaMemcpy(hostData,
                       devData,
                       n * sizeof(T),
                       cudaMemcpyDeviceToHost));
  
  size_t i;
  for(i = 0; i < n; i++) {
    hostData[i] = rand_min + hostData[i] % (rand_max - rand_min);
  }

  CUDA_RUNTIME_FN(cudaMemcpy(devData,
                             hostData,
                             n * sizeof(T),
                             cudaMemcpyHostToDevice));

  /* Cleanup */
  CURAND_CALL(curandDestroyGenerator(gen));

  free(hostData);

  return devData;
}


__global__ void branch_kernel0(unsigned int* iterations) {
  int thread = blockDim.x * blockIdx.x + threadIdx.x;
  
  printf("\n\ngridDim.x: %i, blockIdx.x: %i, blockDim.x: %i, threadIdx.x: %i, thread: %i\n"
         "Kernel value [%" PRId32  "]: %" PRIu64 " \n",
         gridDim.x,
         blockIdx.x,
         blockDim.x,
         threadIdx.x,
         thread,
         thread,
         (size_t) iterations[thread]);

  volatile unsigned int num_iterations = iterations[thread];

  if ((num_iterations & 0x1) == 0) {
    num_iterations = num_iterations << 1;   
  }

  volatile unsigned int i = 0;
  
  while (i < num_iterations) {
    i++;
  }
}

int main() {
  size_t n = 30;
  unsigned int a = 5;
  unsigned int b = 50;
  unsigned int* device_iterations = dev_gen_random(n, a, b);

  size_t nblk = 2;
  
  dim3 Dg(nblk, 1, 1);
  dim3 Db(n / nblk, 1, 1);
  
  branch_kernel0<<<Dg, Db>>>(device_iterations);
  
  CUDA_RUNTIME_FN(cudaFree(device_iterations));

  return EXIT_SUCCESS;
}
