#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

typedef float (*dot_fn_t)(float* a, float* b, int N);

time_t mtime() {
  struct timeval tt = {0};
  gettimeofday(&tt, NULL);
  return tt.tv_sec * 1e6 + tt.tv_usec;
}

float absf(float x) {
  return x < 0.0f ? -x : x;
}

float serial_dot_prod(float* a, float* b, int N) 
{
  float sum = 0.0;
  int i;
  
  for(i=0; i<N; i++) {
    sum += a[i] * b[i];
  }

  return sum;
}

float parallel_dot_prod1(float* a, float* b, int N) {
  float sum = 0.0f;
  int i;
  
#pragma omp parallel for schedule(static, 12) reduction(+:sum)
  for (i = 0; i < N; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float parallel_dot_prod2(float* a, float* b, int N) {
  float sum = 0.0f;
  int i;
  
#pragma omp parallel for schedule(static, 12) shared(sum)
  for (i = 0; i < N; ++i) {
#pragma omp critical
    {
      sum += a[i] * b[i];
    }
  }
  return sum;
}

void fill(float* a, int N) {
  srand(time(NULL));
  for (int i = 0; i < N; ++i) {
    a[i] = (float)(rand() % 0xFFFF);
  }
}

void run(const char* title, float* a, float* b, int N, dot_fn_t f) {
  fill(a, N);
  fill(b, N);

  float serial = serial_dot_prod(a, b, N);
  
  printf("=====\nbegin %s\n=====\n", title);
  
  time_t start = mtime();
  time_t end = mtime();

  float para = f(a, b, N);

  if (para == serial) {
    puts("Correct");
  } else {
    puts("Incorrect");
  }

  printf("serial:\t\t%f\npara:\t\t%f\n", serial, para);
  printf("%s\n", "=====\nend\n=====");
}

int main(int argc, char** argv) {
  int N = 100;
  
  float* a = malloc(sizeof(float) * N);
  assert(a != NULL);
	
  float* b = malloc(sizeof(float) * N);
  assert(b != NULL);

  run("reduction", a, b, N, parallel_dot_prod1);
  run("critical", a, b, N, parallel_dot_prod2); 
	
  // Create an array of floats a and fill it with N random numbers
  // Create an array of floats b and fill it with N random numbers
  // Call serial_dot_prod(a, b, N) and store results in a temporary array
  // Implement a parallel_dot_prod(a, b, N) using OpenMP
  // Call parallel_dot_prod(a, b, N) and store results in another temporary array
  // Compare the two results. If the resutls match, output "Correct" or elsee output "Incorrect"

  return 0;
}
