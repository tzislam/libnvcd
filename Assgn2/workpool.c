#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>
#include <setjmp.h>
#include <execinfo.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <inttypes.h>

// NOTE
//
// There's a fair amount of calls to things like "writef",
// or printf calls. writef writes out to its own file, is buffered,
// and is a macro that's stubbed out by default (unless LOG_INFO is defined,
//                                               nothing will happen).
//
// The user will only see the single desired message indicating the number
// of processes captured, or fatal error messages.
//

//
// Error checking
//

void assert_impl(int cond, const char* expr, const char* file, int line) {
	if (!cond) {
		char msg[256] = {0};
		sprintf(msg, "(%s|%i->%s)", file, line, expr);
		printf("ASSERT FAILURE: %s\n", msg); 
		MPI_Abort(MPI_COMM_WORLD, 0);
	}
}

#define _Assert(cond) assert_impl(cond, #cond, __FILE__, __LINE__)

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

//
// Constants
//

#define TAG 123
#define QNULL -1

enum {
	ABORT = 0,
	ACK = 1,
	REQ_WORK = 2,
	NO_WORK = 3,

	RANDV_MIN = 16,
	RANDV_MAX = 2048
};

//
// Queue
//

int dequeue_buffer(int* buffer, int size) {
	int b = buffer[0];
	int i = 0;
	
	while (i < (size - 1)) {
		buffer[i] = buffer[i + 1];
		buffer[i + 1] = QNULL;
		i++;
	}

	return b;
}

void enqueue_buffer(int* buffer, int size, int length, int value) {
	_Assert(length < size);
	buffer[length] = value;
}

//
// Misc util
//

int64_t get_time() {
	struct timespec tms = {0};
	
	clock_gettime(CLOCK_REALTIME, &tms);

	int64_t micro = tms.tv_sec * 1000000;
	micro += tms.tv_nsec / 1000;

	return micro;
}

time_t get_time_sec() {
	return time(NULL);
}

void randseed() {
	uint32_t tt = (uint32_t)(get_time() & 0xFFFFFFFF);
	srand((unsigned int) tt);
}

int randv() {
	randseed();
	return RANDV_MIN + rand() % (RANDV_MAX - RANDV_MIN);
}

int rand_next(int min, int max) {
	randseed();
	return min + rand() % (max - min);
}

void send_int_nb(int* value, int dest) {
	MPI_Request req;

	MPICALL(MPI_Isend(value,
										1,
										MPI_INT,
										dest,
										TAG,
										MPI_COMM_WORLD,
										&req));

	MPICALL(MPI_Wait(&req,
									 MPI_STATUS_IGNORE));
	
	_Assert(req == MPI_REQUEST_NULL);
}

//
// Used for debugging
//

// Dump each consumer process result
// and compare the sum with stdout (the comparison
// is performed manually, this just makes the data
// easier to access)
void dump_rank_consume_count(int rank, int consume_count) {	
	char buffer[256] = { 0 };
	sprintf(buffer, "%i\n", consume_count);

	char fname[64] = { 0 };
	sprintf(fname, "./rank_%i.log", rank);

	FILE* f = fopen(fname, "wb");
	_Assert(f != NULL);

	fprintf(f, "%s", buffer);

	fclose(f);
}

#ifdef LOG_INFO

static int __rank = 0;

#define WRITEFBUFLEN 1024

static char __writef_buffer[WRITEFBUFLEN + 1024] = { 0 };
static int __writef_counter = 0;

void __writef(int flush, int rank, const char* func, int line, char* fmt, ...) {
	{
		int64_t micro = get_time();
	
		
		int k = sprintf(&__writef_buffer[__writef_counter],
										"[%" PRId64 "]: %s %i|",
										micro,
										func,
										line);

		__writef_counter += k;
		
		int l = 0;

		{
			va_list arg;
			va_start(arg, fmt);
			l = vsprintf(&__writef_buffer[__writef_counter], fmt, arg);
			va_end(arg);
		}
		
		__writef_counter += l;

		if (__writef_counter < WRITEFBUFLEN) {
			strcat(__writef_buffer, "\n");
			__writef_counter++;
		}
	}

	if (__writef_counter >= WRITEFBUFLEN || flush) {
		char fname[512] = { 0 };

		sprintf(fname, "mpif_%i.log", rank);
	
		FILE* f = fopen(fname, "ab+");
		_Assert(f != NULL);
		fprintf(f, "%s\n", __writef_buffer);
		fclose(f);

		__writef_counter = 0;
		memset(__writef_buffer, 0, sizeof(__writef_buffer));
	}
}

#define writef(fmt, ...) __writef(0, __rank, __func__, __LINE__, fmt, __VA_ARGS__)
#define writef_flush(fmt, ...) __writef(1, __rank, __func__, __LINE__, fmt, __VA_ARGS__)

#else

#define writef(fmt, ...) 
#define writef_flush(fmt, ...)

#endif // LOG_INFO


void workpool(int size, int rank, time_t limit) {
	time_t start = get_time_sec();
	time_t elapsed = 0;

	int consumed_counter = 0;

	const int RANK_ROOT = 0;

	writef_flush("size: %i, rank: %i, limit: %li\n",
							 size,
							 rank,
							 limit);

	int abort = 0;
	
	while (elapsed < limit && !abort) {
		writef_flush("[%i] iteration start. elapsed: %li\n", rank, elapsed);
		
		int v = randv();
		
		int dest = rank;
		while (dest == rank) {
			// range is [0, size)
			dest = rand_next(0, size);
		}

		// send off 'v' to 'dest'
		{
			MPI_Request req = MPI_REQUEST_NULL;
			
			MPICALL(MPI_Isend(&v,
												1,
												MPI_INT,
												dest,
												TAG,
												MPI_COMM_WORLD,
												&req));

			MPICALL(MPI_Wait(&req, MPI_STATUS_IGNORE));

			writef_flush("[%i] sending %i to %i\n",
						 rank,
						 v,
						 dest);
		}

		int ack_received = 0;
		while (!ack_received) {
			writef_flush("[%i] ack iteration start\n", rank);
			
			int result = -1;
			MPI_Status status = { 0 };
			
			{
				MPI_Request req = MPI_REQUEST_NULL;
			
				MPICALL(MPI_Irecv(&result,
													1,
													MPI_INT,
													MPI_ANY_SOURCE,
													TAG,
													MPI_COMM_WORLD,
													&req));
			
				MPICALL(MPI_Wait(&req, &status));

				_Assert(req == MPI_REQUEST_NULL);
			}

			switch (result) {
			case ABORT:
				abort = 1;
				ack_received = 1; // ensure we leave the loop
				break;
				
			case ACK:
				ack_received = 1;
				writef_flush("[%i] ack received\n", rank);
				break;
			default: {
				_Assert(RANDV_MIN <= result && result < RANDV_MAX);

				int send_ack = ACK;
				MPI_Request req = MPI_REQUEST_NULL;
			
				MPICALL(MPI_Isend(&send_ack,
													1,
													MPI_INT,
													status.MPI_SOURCE,
													TAG,
													MPI_COMM_WORLD,
													&req));

				MPICALL(MPI_Wait(&req, MPI_STATUS_IGNORE));

				_Assert(req == MPI_REQUEST_NULL);

				consumed_counter++;

				writef_flush("[%i] counter: %i\n", rank, consumed_counter);
			}
			}
		}
		
		elapsed = get_time_sec() - start;
	}

	// ensure all other processes know it's time to abort
	{
		int msg = ABORT;
	
		for (int i = 0; i < size; ++i) {
			MPI_Request req = MPI_REQUEST_NULL;
			
			MPICALL(MPI_Isend(&msg,
												1,
												MPI_INT,
												i,
												TAG,
												MPI_COMM_WORLD,
												&req));

			MPICALL(MPI_Wait(&req, MPI_STATUS_IGNORE));
		}
	}

	// used for sum verification
	dump_rank_consume_count(rank, consumed_counter);
	
	int* buffer = NULL;

	if (rank == RANK_ROOT) {
		buffer = malloc(sizeof(*buffer) * size);

		_Assert(buffer != NULL);
		
		memset(buffer, 0, sizeof(*buffer) * size);
	}
	
	// Gather consume count
	MPICALL(MPI_Gather(&consumed_counter,
										 1,
										 MPI_INT,
										 buffer,
										 1,
										 MPI_INT,
										 RANK_ROOT,
										 MPI_COMM_WORLD));

	if (rank == RANK_ROOT) {
		int total = 0;
		
		for (int i = 0; i < size; ++i) {
			total += buffer[i];
		}
		
		printf("Total number of messages consumed: %i\n", total);

		free(buffer);
	}
} 

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
	
	int rank, size;
	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	writef_flush("size: %i", size);

#ifdef LOG_INFO
	__rank = rank;
#endif
	
	_Assert(size == 4 ||
					size == 8 ||
					size == 12 ||
					size == 16);

	time_t limit = strtol(argv[1], NULL, 10);
	//_Assert(limit == 1);

	printf("limit: %li\n", limit);
	
	workpool(size, rank, limit);
	
	writef_flush("%s", "hit barrier");
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Finalize();

	return 0;
}
