#include <stdio.h>
#include <mpi.h>

void send_recv(int rank, int nproc) {
	const int tag = 1;

	char buffer[4096] = {0};
	size_t offset = 0;
	
	int rank_odd = rank & 1;
	
	for (int i = 0; i < nproc; ++i) {
		if (i != rank && rank_odd != (i & 0x1)) {
			MPI_Send(&rank, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
		}
	}

	for (int i = 0; i < nproc; ++i) {
		if (i != rank && rank_odd != (i & 0x1)) {
			int r = 0;
			MPI_Status s;
			MPI_Recv(&r, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &s);
			offset += sprintf(buffer + offset, "%i, ", r);
		}
	}

	printf("from %i: { %s }\n", rank, buffer);
}

int main(int argc, char** argv) {
	int rank, nproc;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	send_recv(rank, nproc);
	
	//Uncomment this when you have the "rank" variable initialized:
	printf("Hello World from process %d\n", rank);
	
	MPI_Finalize();
	
	return 0;
}
