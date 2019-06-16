/* This program uses the host CURAND API to generate 100
 * pseudorandom floats.
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand.h>

#include <nvcd/nvcd.cuh>

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
      printf("Error at %s:%d\n",__FILE__,__LINE__);            \
return EXIT_FAILURE;}} while(0)

__global__ void branch_kernel_bad(int* iterations) {
  
}

int main() {
  size_t n = 100;
  size_t i;

  curandGenerator_t gen;
  unsigned int *devData, *hostData;
  
  /* Allocate n ints on host */
  hostData = (unsigned int *)calloc(n, sizeof(unsigned int));

  /* Allocate n ints on device */
  CUDA_RUNTIME_FN(cudaMalloc((void **)&devData, n * sizeof(unsigned int)));

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
                       n * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));

  /* Set seed */
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

  /* Generate n floats on device */
  CURAND_CALL(curandGenerate(gen,
                             devData,
                             n));

  /* Copy device memory to host */
  CUDA_RUNTIME_FN(cudaMemcpy(hostData,
                       devData,
                       n * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));
  
  /* Show result */
  for(i = 0; i < n; i++) {
    printf("Value [%" PRIu64  "]: %" PRIu32 "\n", i, hostData[i]);
  }

  printf("%s", "\n");

  /* Cleanup */
  CURAND_CALL(curandDestroyGenerator(gen));
  CUDA_RUNTIME_FN(cudaFree(devData));

  free(hostData);

  return EXIT_SUCCESS;
}
