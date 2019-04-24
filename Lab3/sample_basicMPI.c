#include <stdio.h>
#include <mpi.h>

int main( int argc, char *argv[] ) {
	int nproc, rank;
	// Initialize MPI
	// Get the number of processes in the MPI_COMM_WORLD communicator
	// Get the rank of the current process
	//

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	//Uncomment this when you have the "rank" variable initialized:
	printf("Hello World from process %d\n", rank);

	MPI_Finalize();
	
	//Finalize MPI
	return 0;
}

