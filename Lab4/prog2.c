#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

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

void sendi(int i, int dest, int tag) {
  MPI_Send(&i, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
}

int recvi(int src, int tag) {
  int r;
  MPI_Status s;
  MPI_Recv(&r, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &s);
  return r;
}

int main(int argc, char** argv) {
  int rank, nproc;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int tag = 123;
  
  if (rank == 0) {
    int r = rand() % 100;

    printf("[rank = %i] sending %i to %i\n", rank, r, rank + 1);
    sendi(r, rank + 1, tag);
    
    int q = recvi(nproc - 1, tag);
    printf("[rank = %i] received %i from %i\n", rank, q, nproc - 1);
  } else {
    int r = recvi(rank - 1, tag);
    printf("[rank = %i] received %i from %i\n", rank, r, rank - 1);
    printf("[rank = %i] sending %i to %i\n", rank, r, (rank + 1) % nproc);
    sendi(r, (rank + 1) % nproc, tag);
  }
  
  //Uncomment this when you have the "rank" variable initialized:
  printf("Hello World from process %d\n", rank);
  
  MPI_Finalize();
  
  return 0;
}
