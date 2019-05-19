/* noncontiguous access with a single collective I/O function */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#define FILESIZE      1048576
//#define FILESIZE 1024
#define FILESIZE 20
//#define INTS_PER_BLK
#define INTS_PER_BLK  3

int main(int argc, char **argv)
{
  int *buf, rank, nprocs, nints, bufsize;
  MPI_File fh;
  MPI_Datatype filetype;
	MPI_Status s;

  // Initialize MPI, get rank, get nprocs
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
  bufsize = FILESIZE/nprocs;
  buf = (int *) malloc(bufsize);
  nints = bufsize/sizeof(int);
  memset(buf, 'A'+rank, nints * sizeof(int));

  // Open file in WRONLY mode, make sure to have CREATE flag on
  MPI_File_open(MPI_COMM_WORLD,
								"outputfile",
								MPI_MODE_CREATE | MPI_MODE_WRONLY,
								MPI_INFO_NULL, &fh);
  // Declare type vector
	MPI_Type_vector(1, INTS_PER_BLK, nprocs, MPI_INT, &filetype);

	// Commit newly declared type vector
	MPI_Type_commit(&filetype);
	
	// Set file view
	MPI_File_set_view(fh,
										INTS_PER_BLK * rank * sizeof(int),
										MPI_INT,
										filetype,
										"native",
										MPI_INFO_NULL);

	MPI_File_write_all(fh, buf, nints, MPI_INT, &s);

	MPI_File_close(&fh);

	MPI_Type_free(&filetype);
	
	// Write all
	// close file
	// Type free
  free(buf);
    
// MPI finalize
	MPI_Finalize();
  return 0; 
}
