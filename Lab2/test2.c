#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

float serial_dot_prod(float* a, float* b, int N) 
{
  float sum = 0.0;
  int i;
  for(int i=0; i<N; i++) {
     sum += a[i] * b[i];
  }
  return sum;
}

float parallel_dot_prod1(float* a, float* b, int N) {
  float sum = 0.0f;
  int i;
  
#pragma omp parallel for reduction(+:sum)
  {
    for (i = 0; i < N; ++i) {
      sum += a[i] * b[i];
    }
  }
  return sum;
}

int main(int argc, char** argv) {
	int N = 100;
	int* m = malloc(sizeof(int) * N);
	assert(m != NULL);

	
	
	// Create an array of floats a and fill it with N random numbers
	// Create an array of floats b and fill it with N random numbers
	// Call serial_dot_prod(a, b, N) and store results in a temporary array
	// Implement a parallel_dot_prod(a, b, N) using OpenMP
	// Call parallel_dot_prod(a, b, N) and store results in another temporary array
	// Compare the two results. If the resutls match, output "Correct" or elsee output "Incorrect"

	return 0;
}
