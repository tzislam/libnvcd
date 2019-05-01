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
	MPI_Init(&argc, &argv);
	
	int world_rank;
	int world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	// TODO: Get rank and size of the communicator
 
	// Seed the random number generator to get different results each time
	srand(time(NULL) * world_rank);

	int rand_num = rand() % 100;
	int rank;

	printf("world_rank: %i, rand_num: %i\n", world_rank, rand_num);

	TMPI_Rank(&rand_num, &rank, MPI_INT, MPI_COMM_WORLD);

	printf("Rank for %i on process %i - %i\n", rand_num, world_rank, rank);

	MPI_Barrier(MPI_COMM_WORLD);
	// TODO: Use a barrier t make sure all processes come here before finishing.
	// TODO: Finalize

	MPI_Finalize();

	return 0;
}
