#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <stdint.h>

// file is changed 

typedef double scalar_t;

#define SCALAR_FMT "f"

typedef scalar_t (*dot_fn_t)(scalar_t* a, scalar_t* b, int N);

time_t mtime() {
	struct timeval tt = {0};
	gettimeofday(&tt, NULL);
	return tt.tv_sec * 1e6 + tt.tv_usec;
}

scalar_t serial_dot_prod(scalar_t* a, scalar_t* b, int N) {
	scalar_t sum = (scalar_t)0;
	int i;
  
	for(i=0; i<N; i++) {
		sum += a[i] * b[i];
	}

	return sum;
}

scalar_t parallel_dot_prod(scalar_t* a, scalar_t* b, int N) {
	scalar_t sum = 0.0f;
	int i;

#ifdef USE_CRITICAL

#pragma message "Path chosen: critical"

#pragma omp parallel for shared(sum)
	for (int i = 0; i < N; i++) {
#pragma omp critical
		{
			sum += a[i] * b[i];
		}
	}
#else
	
#pragma message "Path chosen: reduction"
	
#pragma omp parallel for reduction(+:sum) schedule(static, 1)
	for (i = 0; i < N; i++) {
		sum += a[i] * b[i];
	}

#endif	// USE_CRITICAL
	return sum;
}

void fill(scalar_t* a, int N) {
	srand(time(NULL));
	for (int i = 0; i < N; ++i) {
		a[i] = (scalar_t)(rand() % 0xFFFF);
	}
}

int run(scalar_t* a, scalar_t* b, int N, dot_fn_t f) {
	fill(a, N);
	fill(b, N);

	scalar_t serial = serial_dot_prod(a, b, N);

	scalar_t para = f(a, b, N);

	int ret = para == serial;
	
	if (ret) {
		puts("Correct");
	} else {
		puts("Incorrect");
	}

	printf("\tserial:\t\t%" SCALAR_FMT "\n\tpara:\t\t%" SCALAR_FMT "\n", serial, para);

	return ret;
}

int main(int argc, char** argv) {
	int N = 100;
  
	scalar_t* a = malloc(sizeof(scalar_t) * N);
	assert(a != NULL);
	
	scalar_t* b = malloc(sizeof(scalar_t) * N);
	assert(b != NULL);

	return run(a, b, N, parallel_dot_prod) == 1 ? 0 : 1;
	
	// Create an array of floats a and fill it with N random numbers
	// Create an array of floats b and fill it with N random numbers
	// Call serial_dot_prod(a, b, N) and store results in a temporary array
	// Implement a parallel_dot_prod(a, b, N) using OpenMP
	// Call parallel_dot_prod(a, b, N) and store results in another temporary array
	// Compare the two results. If the resutls match, output "Correct" or elsee output "Incorrect"
}
