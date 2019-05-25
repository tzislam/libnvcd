/* noncontiguous access with a single collective I/O function */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fileinfo.h"
#include <assert.h>

void mpicall_impl(int error_code, const char* expr, int line) {
  /*
   * shamelessly stolen boilerplate: 
   * https://computing.llnl.gov/tutorials/mpi/errorHandlers.pdf
   */
  if (error_code != MPI_SUCCESS) {
    char error_string[4096] = { 0 };
    int length_of_error_string = 0;
    int error_class = 0;
    int my_rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    MPI_Error_class(error_code, &error_class);
    MPI_Error_string(error_class, error_string, &length_of_error_string);

    fprintf(stderr, "ERROR (line %i): %s\n", line, expr);
    
    fprintf(stderr, "\t%3d: %s\n", my_rank, error_string);
    MPI_Error_string(error_code, error_string, &length_of_error_string);

    fprintf(stderr, "\t%3d: %s\n", my_rank, error_string);
    MPI_Abort(MPI_COMM_WORLD, error_code);
  }
}

#define MPICALL(expr) mpicall_impl(expr, #expr, __LINE__)

#define _N 2
typedef struct space {
  int sizes[_N];
  int subsizes[_N];
  int starts[_N];
} space_t;

int main(int argc, char **argv)
{
  mpi_type_t* buf = NULL;
  int rank, nprocs, nvals, bufsize;//
  MPI_File fh;
  MPI_Datatype filetype;
  MPI_Status s;
  
  // Initialize MPI, get rank, get nprocs
  MPICALL(MPI_Init(&argc, &argv));

  MPICALL(MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
  
  MPICALL(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
  MPICALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  int type_size = 0;
  MPICALL(MPI_Type_size(MPI_TDEF_TYPE, &type_size));

  printf("type size: %i\n", type_size);
  
  bufsize = (FILESIZE/nprocs);
  buf = (mpi_type_t *) malloc(bufsize);
  assert(buf != NULL);

  nvals = bufsize/type_size;
  memset(buf, 0, bufsize);

  // Open file in RDONLY mode
  MPI_File_open(MPI_COMM_WORLD,
                "outputfile",
                MPI_MODE_RDONLY,
                MPI_INFO_NULL, &fh);
  // Declare type vector
  //MPICALL(MPI_Type_vector(nvals/VALS_PER_BLK, VALS_PER_BLK, nprocs * VALS_PER_BLK, MPI_TDEF_TYPE, &filetype));

  {
    space_t s;
    
    s.sizes[0] = nvals / VALS_PER_BLK;
    s.subsizes[0] = nvals / VALS_PER_BLK;
    s.starts[0] = 0;

    s.sizes[1] = nprocs * VALS_PER_BLK;
    s.subsizes[1] = VALS_PER_BLK;
    s.starts[1] = rank * VALS_PER_BLK;

    space_t* r = NULL;

    if (rank == 0) {
      r = calloc(nprocs, sizeof(space_t));
    }

    MPI_Gather(&s, sizeof(s), MPI_CHAR, r, sizeof(s), MPI_CHAR, 0, MPI_COMM_WORLD);

#define line(x) "\t" #x " = %i\n"
    
    if (rank == 0) {
      for (int i = 0; i < nprocs; ++i) {    
        printf("[rank %i]\n"
               line(r[i].sizes[0])
               line(r[i].subsizes[0])
               line(r[i].starts[0])
               
               line(r[i].sizes[1])
               line(r[i].subsizes[1])
               line(r[i].starts[1]),
               i,
               r[i].sizes[0], r[i].subsizes[0], r[i].starts[0],
               r[i].sizes[1], r[i].subsizes[1], r[i].starts[1]);
      }
    }
    
    MPICALL(MPI_Type_create_subarray(_N,
                                     s.sizes,
                                     s.subsizes,
                                     s.starts,
                                     MPI_ORDER_C,
                                     MPI_TDEF_TYPE,
                                     &filetype));
  }                      
  
  // Commit newly declared type vector
  MPICALL(MPI_Type_commit(&filetype));
  
  // Set file view
  MPI_File_set_view(fh,
                    0,
                    MPI_TDEF_TYPE,
                    filetype,
                    "native",
                    MPI_INFO_NULL);

  int* readbuf = malloc(bufsize * 4);
  assert(readbuf != NULL);
  memset(readbuf, 0, bufsize * 4);
  
  MPICALL(MPI_File_read_all(fh, readbuf, nvals * 2, MPI_TDEF_TYPE, &s));

  MPICALL(MPI_File_close(&fh));

  for (int i = 0; i < nvals; ++i) {
    char b[sizeof(mpi_type_t) + 1] = {0};
    memcpy(b, &readbuf[i], sizeof(mpi_type_t));
    
    printf("[%i][%i] %s | 0x%x | %i\n", rank, i, b, readbuf[i], readbuf[i]);
  }
  
  
  MPICALL(MPI_Type_free(&filetype));
  
  // Write all
  // close file
  // Type free
  free(buf);
  free(readbuf);
    
  // MPI finalize
  MPICALL(MPI_Finalize());
  return 0; 
}


