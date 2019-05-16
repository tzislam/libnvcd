#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mpi.h"
#define MAXPROC 100    /* Max number of procsses */
#define MAXITER 100000000
void compute_something(){
    // This function is basically a do nothing function. But in a practical scenario,
    // can be used to do useful computation
    /* While the messages are delivered, we could do computations here */
    printf("I am a very busy processor.\n");
    int i;
    for(i = 0; i < MAXITER; i++)
        ;
}

int main(int argc, char* argv[]) {
  int i, nproc, rank, index;
  const int tag  = 42;    /* Tag value for communication */
  const int root = 0;     /* Root process in broadcast */

  MPI_Status status;              /* Status object for non-blocing receive */
  MPI_Request recv_req[MAXPROC];  /* Request objects for non-blocking receive */
  
  char hostname[MAXPROC][MPI_MAX_PROCESSOR_NAME];  /* Received host names */
  char myname[MPI_MAX_PROCESSOR_NAME]; /*local host name string */
  int namelen; // Length of the name

  memset(hostname, 0, sizeof(hostname));
  
 //Begin parallel region
 //Init
 //Get rank
 //Get group size

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  
  // Get hostname
  //MPI_Get_processor_name --> myname, namelen
  
  MPI_Get_processor_name(myname, &namelen);
  myname[namelen] = '\0';

  printf("HostName for Rank = %i: %s\n", rank, myname);

  /* Terminate received buffer with null byte */
  if (rank == 0) {    /* Process 0 does this */
	  
    /* Send a broadcast message containing its rank to all other processes */
	  MPI_Bcast(&rank, 1, MPI_INT, root, MPI_COMM_WORLD);
	  
	  /* Start non-blocking calls to receive messages from all other processes */
		/* Move on to computation*/
		compute_something();
    
    /* Wait for the receive calls to finish.
			 MPI_Waitall
			 Iterate to receive messages from all other processes and print their hostnames*/
	  for (int i = 1; i < nproc; ++i) {
		  MPI_Irecv(hostname[i],
								MPI_MAX_PROCESSOR_NAME - 1,
								MPI_CHAR,
								i,
								tag,
								MPI_COMM_WORLD,
								&recv_req[i]);
	  }

		for (int i = 1; i < nproc; ++i) {
			int k = 0;
			MPI_Status s;
			
			MPI_Waitany(nproc - 1,
									&recv_req[1],
									&k,
									&s);
			
			printf("Irecv k = %i, hostname[k] = %s\n",
						 k + 1,
						 hostname[k + 1]);
		}
  } 
  else { /* all other processes do this */
	  int result = 0;
	  MPI_Bcast(&result, 1, MPI_INT, root, MPI_COMM_WORLD);

	  printf("[Rank = %i] Received %i\n", rank, result);

	  
    /* Receive the broadcasted message from process 0 using a blocking call */
      
    /* Send local hostname to process 0 using a non blocking send */
	  MPI_Request r;
	  MPI_Isend(myname, namelen, MPI_CHAR, root, tag, MPI_COMM_WORLD, &r);
	  
		// Then move on to computation
		compute_something();
		/* Wait for the MPI_Isent to finish by calling MPI Wait*/

	  MPI_Status s;
	  MPI_Wait(&r, &s);
  }

 // Finish by finalizing the MPI library
  MPI_Finalize();
  exit(0);
}
