#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdint.h>

// compiler output laziness
#define PRG(message) _Pragma(#message)
#define M(x) PRG(message "Data type chosen: " #x)

// options
#define SCALAR_FLOAT 0
#define SCALAR_DOUBLE 1
#define SCALAR_UINT 2

#define METHOD_CRITICAL 0
#define METHOD_REDUCTION 1

// defaults
#ifndef NUM_SCALARS
#pragma message "Defaulting to 1000 values per vector"
#define NUM_SCALARS 1000
#endif

#ifndef SCALAR_TYPE
#define SCALAR_TYPE SCALAR_FLOAT
#endif

#ifndef METHOD_TYPE
#define METHOD_TYPE METHOD_CRITICAL
#endif

// set meta parameters depending on type used

#if SCALAR_TYPE == SCALAR_FLOAT

M(float)

typedef float scalar_t;
#define SCALAR_FMT "f"

#elif TYPE == SCALAR_DOUBLE

M(double)

typedef double scalar_t;
#define SCALAR_FMT "f"

#elif TYPE == SCALAR_UINT

M(unsigned int)

typedef unsigned int scalar_t;
#define SCALAR_FMT "u"

#endif

//
// functions
//

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
  
#if METHOD_TYPE == METHOD_CRITICAL

#pragma message "Parallel method: critical"
#pragma omp parallel for shared(sum)
  
  for (i = 0; i < N; i++) {
#pragma omp critical
    {
      sum += a[i] * b[i];
    }
  }

#else
  
#pragma message "Parallel method: reduction"  
#pragma omp parallel for reduction(+:sum)
  for (i = 0; i < N; i++) {
    sum += a[i] * b[i];
  }

#endif
  return sum;
}

void fill(scalar_t* a, int N) {
  srand(time(NULL));
  for (int i = 0; i < N; ++i) {
    int x = rand();
    
    a[i] = (scalar_t)(x);
  }
}

int cmp(scalar_t a, scalar_t b) {
  // Something like this is necessary else
  // the comparison will always fail, at least
  // with unbounded random values
#if SCALAR_TYPE == SCALAR_FLOAT
  float absa = a < 0.0f ? -a : a;
  float absb = b < 0.0f ? -b : b;
  float epsilon = 0.000001f; // lowest epsilon that can be used
  float diff = a - b;
  float absdiff = diff < 0.0f ? -diff : diff;

  float rel_err = (absdiff / (absa + absb)); 

#ifdef DEBUG
  printf("Relative error: %f | Epsilon: %f\n", rel_err, epsilon);
#endif

  return rel_err < epsilon;
  
#else
  return a == b;
#endif
}

int run(scalar_t* a, scalar_t* b, int N) {
  fill(a, N);
  fill(b, N);

  scalar_t serial = (scalar_t)0;
  scalar_t para = (scalar_t)0;
  
  double stime = 0.0;
  {
    double s = omp_get_wtime();
    serial = serial_dot_prod(a, b, N);
    double e = omp_get_wtime();

    stime = e - s;
  }

  double ptime = 0.0;
  {
    double s = omp_get_wtime();
    para = parallel_dot_prod(a, b, N);
    double e = omp_get_wtime();

    ptime = e - s;
  }
  
  int ret = cmp(para, serial);

  const char* retstr = ret ? "Correct" : "Incorrect";
  
  printf("%s. Serial_time: %f, Parallel_time: %f\n", retstr, stime, ptime);

#ifdef DEBUG
  printf("\tserial:\t\t%" SCALAR_FMT "\n\tpara:\t\t%" SCALAR_FMT "\n", serial, para);
#endif

  return ret;
}

int main(int argc, char** argv) {
  int N = NUM_SCALARS;
  
  scalar_t* a = malloc(sizeof(scalar_t) * N);
  assert(a != NULL);
  
  scalar_t* b = malloc(sizeof(scalar_t) * N);
  assert(b != NULL);

  return run(a, b, N) == 1 ? 0 : 1;
  
  // Create an array of floats a and fill it with N random numbers
  // Create an array of floats b and fill it with N random numbers
  // Call serial_dot_prod(a, b, N) and store results in a temporary array
  // Implement a parallel_dot_prod(a, b, N) using OpenMP
  // Call parallel_dot_prod(a, b, N) and store results in another temporary array
  // Compare the two results. If the resutls match, output "Correct" or elsee output "Incorrect"
}
