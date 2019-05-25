#ifndef __FILEINFO_H__
#define __FILEINFO_H__

//#define FILESIZE      1048576
//#define FILESIZE 1024

#ifndef LAYOUT
#define LAYOUT 0
#endif

#if LAYOUT == 0
#define FILESIZE 128
//#define INTS_PER_BLK
#define VALS_PER_BLK  4
#elif LAYOUT == 1
#define FILESIZE 128
#define VALS_PER_BLK 2
#elif LAYOUT == 2
#define FILESIZE 32
#define VALS_PER_BLK 1
#endif


typedef int mpi_type_t;

#define MPI_TDEF_TYPE MPI_INT

#define MPI_TDEF_FMT "s"

#endif //__FILEINFO_H__
