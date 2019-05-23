#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "perf_dump.h"
/* Define length of dot product vectors and number of OpenMP threads */
#define VECLEN 100
#define NUMTHREADS 24

int main (int argc, char* argv[])
{
  int i, myid, tid, numprocs, len=VECLEN, threads=NUMTHREADS;
  double *a, *b;
  double mysum, allsum, sum, psum;

  /* MPI Initialization */
  MPI_Init (&argc, &argv);
  pdump_init();
  MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &myid);
  /* 
   *    Each MPI task uses OpenMP to perform the dot product, obtains its partial sum, 
   *       and then calls MPI_Reduce to obtain the global sum.
   *       */
  if (myid == 0)
    printf("Starting omp_dotprod_hybrid. Using %d tasks...\n",numprocs);

  /* Assign storage for dot product vectors */
  a = (double*) malloc (len*threads*sizeof(double));
  b = (double*) malloc (len*threads*sizeof(double));
 
  /* Initialize dot product vectors */
  for (i=0; i<len*threads; i++) {
    a[i]=1.0;
    b[i]=a[i];
  }

  /*
   *    Perform the dot product in an OpenMP parallel region for loop with a sum reduction
   *       For illustration purposes:
   *            - Explicitly sets number of threads
   *                 - Gets and prints number of threads used
   *                      - Each thread keeps track of its partial sum
   *                      */

  /* Initialize OpenMP reduction sum */
  sum = 0.0;

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
  /* Print this task's partial sum */
  mysum = sum;
  printf("Task %d partial sum = %f\n",myid, mysum);

  /* After the dot product, perform a summation of results on each node */
  MPI_Reduce (&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (myid == 0) 
    printf ("Done. Hybrid version: global sum  =  %f \n", allsum);

  free (a);
  free (b);
  pdump_finalize();
  MPI_Finalize();
}
