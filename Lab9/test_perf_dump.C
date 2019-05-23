#include <mpi.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <omp.h>
#include <assert.h>
#include "perf_dump.h"
using namespace std;

#define MAX_STEP 98
#define VEC_SIZE 1000

void init_vector(vector<double>& vec) {
  for (size_t i=0; i < vec.size(); i++) {
    vec[i] = (double) rand();
  }
}


double dot(const vector<double>& a, const vector<double>& b) {
  double sum = 0;
  for (size_t i=0; i < a.size(); i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

int main(int argc, char **argv) {
  int support_provided;
  MPI_Init(&argc, &argv);

  /// ANNOTATION_STEP_1: Call pdump_init() to initialize the library.
  pdump_init();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /// ANOOTATION_STEP_2: Call pdump_start_region() or pdump_start_region_with_name(__func__)
  pdump_start_region_with_name("func1");
  // 1st way to annotate pdump_*_profile()
#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    vector<double> a(VEC_SIZE);
    vector<double> b(VEC_SIZE);
    srand(23578 * (thread_id * rank));
    /// ANNOTATION_STEP_3: Call pdump_start_profile() to turn hardware counters on
    /// right before the "#pragma omp for" call.
    pdump_start_profile();

#pragma omp for
    for (size_t step=0; step < MAX_STEP; step++) {
      init_vector(a);
      init_vector(b);
      double d = dot(a, b);

      double prod=0;
    }// omp_for done
    /// ANNOTATION_STEP_4: Turn hardware counters off.
    pdump_end_profile();
  }
  /// ANNOTATION_STEP_5: Done with this kernel.
  pdump_end_region();

  /// Example of how to profile multiple kernels in the same code.
  pdump_start_region();
#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    vector<double> a(VEC_SIZE);
    vector<double> b(VEC_SIZE);
    srand(23578 * (thread_id * rank));
    pdump_start_profile();
#pragma omp for
    for (size_t step=0; step < MAX_STEP; step++) {

      init_vector(a);
      init_vector(b);
      double d = dot(a, b);

      double prod=0;
      //      MPI_Reduce(&d, &prod, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // PMPI_Reduce fails

      //      if (rank == 0) {
      //        cout << step << ":\t" << prod << endl;
      //      }
    }// omp_for done
    pdump_end_profile();
  }
  pdump_end_region();

  /// ANNOTATION_STEP_6: Finalize all data structures in the library and
  /// dump data. MUST be the last call you make to the library.
  pdump_finalize();
  MPI_Finalize();
}
