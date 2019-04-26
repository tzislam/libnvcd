// Author: Wes Kendall
// Copyright 2013 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// Runs the TMPI_Rank function with random input.
//
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "tmpi_rank.h"
#include <time.h>

int main(int argc, char** argv) {
  //TODO: Initialize MPI

  int world_rank;
  int world_size;
  // TODO: Get rank and size of the communicator

  // Seed the random number generator to get different results each time
  srand(time(NULL) * world_rank);

  int rand_num = rand();
  int rank;
  TMPI_Rank(&rand_num, &rank, MPI_INT, MPI_COMM_WORLD);
  printf("Rank for %f on process %d - %d\n", rand_num, world_rank, rank);

  // TODO: Use a barrier to make sure all processes come here before finishing.
  // TODO: Finalize
}
