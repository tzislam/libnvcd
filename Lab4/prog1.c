#include <stdio.h>

void send_recv(int odd, int rank, int nproc) {
	const tag = 1;

	for (int i = 0; i < nproc; ++i) {
		if (i != rank && odd != (i & 0x1)) {
			MPI_Status err;
			MPI_Send(&rank, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &err);
		}
	}

	for (int i = 0; i < nproc; ++i) {
		if (i != rank && odd != (i & 0x1)) {
			int r = 0;
			MPI_Recv(&r, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &err);
		}
	}
}

int main() {
	int rank, nproc;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int tag = 0;
	
	//Uncomment this when you have the "rank" variable initialized:
	printf("Hello World from process %d\n", rank);

	
	
	MPI_Finalize();
	
	return 0;
}
