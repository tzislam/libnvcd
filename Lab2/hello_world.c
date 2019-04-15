/******************************************************************************
* FILE: omp_hello.c
* DESCRIPTION:
*   OpenMP Example - Hello World - C/C++ Version
*   In this simple example, the master thread forks a parallel region.
*   All threads in the team obtain their unique thread number and print it.
*   The master thread only prints the total number of threads.  Two OpenMP
*   library routines are used to obtain the number of threads and each
*   thread's number.
* AUTHOR: Blaise Barney  5/99
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
  int nthreads, tid;  
  
#pragma omp parallel private(tid)
  {
    tid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
    
    //Open the parallel region
    /* Obtain thread number */
    // tid = call a function to get the thread id.
    printf("Hello World from thread = %d\n", tid);

    if (tid == 0) {
      //Use the proper clause so that only master thread does the following print statement.
      //nthreads = call the function to get the total number of threads
      printf("Number of threads = %d\n", nthreads);
    }
  }
  return 0;
}

