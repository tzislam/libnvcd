/* noncontiguous access with a single collective I/O function */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fileinfo.h"

int main(int argc, char **argv)
{
  mpi_type_t* buf = NULL;
  int rank, nprocs, nvals, bufsize;//
  MPI_File fh;
  MPI_Datatype filetype;
  MPI_Status s;

  // Initialize MPI, get rank, get nprocs
  MPI_Init(&argc, &argv);
  
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int type_size = 0;
  MPI_Type_size(MPI_TDEF_TYPE, &type_size);

  printf("type size: %i\n", type_size);
  
  bufsize = (FILESIZE/nprocs);
  buf = (mpi_type_t *) malloc(bufsize);
  nvals = bufsize/type_size;
  memset(buf, 'A'+rank, bufsize);

  // Open file in WRONLY mode, make sure to have CREATE flag on
  MPI_File_open(MPI_COMM_WORLD,
                "outputfile",
                MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fh);
  // Declare type vector
  MPI_Type_vector(nvals/VALS_PER_BLK, VALS_PER_BLK, nprocs * VALS_PER_BLK, MPI_TDEF_TYPE, &filetype);

  // Commit newly declared type vector
  MPI_Type_commit(&filetype);
  
  // Set file view
  MPI_File_set_view(fh,
                    VALS_PER_BLK * rank * type_size,
                    MPI_TDEF_TYPE,
                    filetype,
                    "native",
                    MPI_INFO_NULL);

  MPI_File_write_all(fh, buf, nvals, MPI_TDEF_TYPE, &s);

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
