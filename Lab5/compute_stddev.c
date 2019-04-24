#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>

// Creates an array of random numbers. Each number has a value from 0 - 1
float *rand_num_gen(int num_elements) {
  float *rand_nums = (float *)malloc(sizeof(float) * num_elements);
  assert(rand_nums != NULL);
  int i;
  for (i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand() / (float)RAND_MAX);
  }
  return rand_nums;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: avg num_elements_per_proc\n");
    exit(1);
  }
  int nproc, rank;
  int num_elements_per_proc = atoi(argv[1]);

  // Create a random array of elements on all processes.
  srand(123456*nproc); // Seed the random number generator of processes uniquely
  float *rand_nums = NULL;
  rand_nums = rand_num_gen(num_elements_per_proc);

  // Sum the numbers locally

  // Reduce all of the local sums into the global sum in order to
  // calculate the mean

  // Compute the local sum of the squared differences from the mean

  // Reduce the global sum of the squared differences to the root process
  // and print off the answer

  // The standard deviation is the square root of the mean of the squared
  // differences.

  // Clean up
  free(rand_nums);

}
