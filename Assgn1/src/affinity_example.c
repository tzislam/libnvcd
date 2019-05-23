#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char **argv){
  //Change the following 2 variables to change input parameter size
  int nlocal = atol(argv[1]);//100000000;
  int nsteps = atol(argv[2]);
  // Allocating memory
  double *in = (double*) malloc( (nlocal+2) * sizeof(double));
  double *out = (double*) malloc(nlocal * sizeof(double));
  for (int i=0; i<nlocal+2; i++)
    in[i] = 1.0;
  for (int i=0; i<nlocal; i++)
    out[i] = 0.0;
  double start_time, end_time;
  for (int step=0; step < nsteps; step++) {
    start_time = omp_get_wtime();
#pragma omp parallel for schedule(static) num_threads(12)
    for (int i=0; i < nlocal; i++) {
      out[i] = ( in[i]+in[i+1]+in[i+2] )/3.;
    }
#pragma omp parallel for schedule(static) num_threads(12)
    for (int i=0; i < nlocal; i++){
      in[i+1] = out[i];
    }
    in[0] = 0;
    in[nlocal+1] = 1;
  }
  end_time = omp_get_wtime();
  printf("%lf\n", end_time - start_time);
}
