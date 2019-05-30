#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "perf_dump.h"

#define VECLEN 100
#define NUMTHREADS 24

int main (int argc, char* argv[])
{
  int i, myid, tid, numprocs, len=VECLEN, threads=NUMTHREADS;
  double *a, *b;
  double mysum, allsum, sum, psum;


  // init mpi and pdump libraries
  MPI_Init (&argc, &argv);

  pdump_init();

  MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &myid);

  // have N processes running concurrently, so it makes sense to
  // only report state in one process.
  if (myid == 0)
    printf("Starting omp_dotprod_hybrid. Using %d tasks...\n",numprocs);

  // heap allocate  24 * 100 matrix, with
  // each thread representing a row in the matrix
  a = (double*) malloc (len*threads*sizeof(double));
  b = (double*) malloc (len*threads*sizeof(double));

  // set all values to 1.0
  for (i=0; i<len*threads; i++) {
    a[i]=1.0;
    b[i]=a[i];
  }

  sum = 0.0;

  // toggle the beginning of a region so we
  // can begin profiling.
  // we want to tid, our loop index, and psum
  // to be private to each thread; the loop itself
  // will be subdivided into length iterations
  // per thread, which is what allows for i
  // to be private in the first place ("num_threads(threads)" parameter ensures this)
  // we continuously update each thread's psum in the loop, so that we know what exactly the partial
  // sum for that thread will be when it's finished. since "sum" is reduced over the '+' operator,
  // we have an implicit private relationship over the sum variable as well, thus preventing
  // psum from sharing values from other threads; we update psum in the loop since
  // outside of the loop the reduced sum variable will have been joined with all other
  // thread sum instances
  pdump_start_region();
#pragma omp parallel private(i,tid,psum) num_threads(threads)
  {
    psum = 0.0;
    tid = omp_get_thread_num();
    if (tid ==0)
    {
      threads = omp_get_num_threads();
      printf("Task %d using %d threads\n",myid, threads);
    }

    // here we begin our profiler
    pdump_start_profile();
#pragma omp for reduction(+:sum)
    for (i=0; i<len*threads; i++)
    {
      sum += (a[i] * b[i]);
      psum = sum;
    }
    pdump_end_profile();

    printf("Task %d thread %d partial sum = %f\n",myid, tid, psum);
  }
  pdump_end_region();

  mysum = sum;
  printf("Task %d partial sum = %f\n",myid, mysum);

  // sum up each sum value and send them all over rank 0, which will store
  // the result in its allsum variable.
  MPI_Reduce (&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (myid == 0) 
    printf ("Done. Hybrid version: global sum  =  %f \n", allsum);

  free (a);
  free (b);
  pdump_finalize();
  MPI_Finalize();
}
