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

#define ST_PRINT_BUF_SZ 4096

#define TAG 123

static void stacktrace() {
#ifdef __linux__
	void* buffer[10] = { NULL };
	int num_written = backtrace(buffer, 10);

	if (num_written > 0) {
		char** syms = backtrace_symbols(buffer, num_written);
		if (syms != NULL) {
			char buffer[ST_PRINT_BUF_SZ] = {0};
			size_t chars_written = 0;

			for (int i = 0; i < num_written; ++i) {
				char tmp[256] = {0};

				chars_written += (size_t) snprintf(
						tmp, 
						sizeof(tmp), 
						"[%i] %s\n", 
						i, 
						syms[i]
						);

				if (chars_written < ST_PRINT_BUF_SZ) {
					strcat(buffer, tmp);
				}
			}

			printf("STACKTRACE\n====\n%s\n====\n", buffer); 

			free(syms);
		} else {
			perror("backtrace_symbols");
		}
	} else {
		/*  
		 * these may be useless: man pages are 
		 * a bit vague wrt backtrace errors 
		 */
		perror("backtrace");
	}
#endif
}

static jmp_buf _jmp_buf;

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

#define QNULL -1

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
	int i = 0;

	_Assert(length < size);
	buffer[length] = value;
}

enum {
	ABORT = 0,
	ACK = 1,
	REQ_WORK = 2,
	NO_WORK = 3,

	RANDV_MIN = 16,
	RANDV_MAX = 2048
};

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

static int __tmp = RANDV_MIN;
int randv() {
	uint32_t tt = (uint32_t)(get_time() & 0xFFFFFFFF);
	
	srand((unsigned int) tt);
	
	return RANDV_MIN + rand() % (RANDV_MAX - RANDV_MIN);
}

static int __rank = 0;

#define WRITEFBUFLEN 1024

static char __writef_buffer[WRITEFBUFLEN + 1024] = { 0 };
static int __writef_counter = 0;

void __writef(int rank, const char* func, int line, char* fmt, ...) {
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

	if (__writef_counter >= WRITEFBUFLEN) {
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

#define writef(fmt, ...) __writef(__rank, __func__, __LINE__, fmt, __VA_ARGS__)

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

void broker(int size, int rank, time_t limit) {
	_Assert(rank == 0);
	time_t start = get_time_sec();
	time_t elapsed = get_time_sec() - start;

	int RESPONSE_ACK = ACK;

	// incremented each time when an abort is sent.
	// once the amount is equal to size,
	// the loop is finished.
	int aborts_sent = 1;
	int consumer_aborts_sent = 0;
	int num_consumers = (size / 2) - 1;
	
	MPI_Request* request_buf = malloc(sizeof(*request_buf) * size);
	int* result_buf = malloc(sizeof(*result_buf) * size);
	int* may_recv_buf = malloc(sizeof(*may_recv_buf) * size);
	int* needs_ack = malloc(sizeof(*needs_ack) * size);
	int* job_q = malloc(sizeof(*job_q) * size);
	int* overflow_q = malloc(sizeof(*overflow_q) * size);
	
	_Assert(request_buf != NULL);
	_Assert(result_buf != NULL);
	_Assert(may_recv_buf != NULL);
	_Assert(needs_ack != NULL);
	_Assert(job_q != NULL);
	_Assert(overflow_q != NULL);

	int job_q_len = 0;
	int overflow_q_len = 0;
	
	for (int i = 1; i < size; ++i) {
		request_buf[i] = MPI_REQUEST_NULL;
		result_buf[i] = -1;
		may_recv_buf[i] = 1;
		needs_ack[i] = 0;
		job_q[i] = QNULL;
		overflow_q[i] = QNULL;
	}
	
	while (aborts_sent < size) {
		writef("%s", "iteration start");

		// Assuming our job_q isn't full,
		// check the overflow_q and transfer
		// any values over
		if (job_q_len < size) {
			while (job_q_len < size && overflow_q_len > 0) {
				int v = dequeue_buffer(overflow_q, size);
				
				enqueue_buffer(job_q, size, job_q_len, v);
				job_q_len++;

				overflow_q_len--;
			}
		}
		
		// handle recv calls; we only
		// modify the receive buffer for the
		// process if we haven't toggled
		// their recv flag off for this iteration
		for (int i = 1; i < size; ++i) {
			writef("may_recv_buf[%i] = %i\n",
						 i,
						 may_recv_buf[i]);
			 
			if (may_recv_buf[i]) {
				if (request_buf[i] == MPI_REQUEST_NULL) {
					MPICALL(MPI_Irecv(&result_buf[i],
														1,
														MPI_INT,
														i,
														TAG,
														MPI_COMM_WORLD,
														&request_buf[i]));
				}
			} else {
				request_buf[i] = MPI_REQUEST_NULL;
			}
		}
		
		int source_rank = -1;
						
		MPICALL(MPI_Waitany(size - 1,
												request_buf + 1,
												&source_rank,
												MPI_STATUS_IGNORE));

		// add one to offset the difference
		source_rank++;
		
		_Assert(request_buf[source_rank] == MPI_REQUEST_NULL);
		_Assert(source_rank != MPI_UNDEFINED);

		// ensure that all flags are toggled on
		// for next iteration (unless the next
		// messaged received deems that one flag
		// must be turned off)
		for (int i = 1; i < size; ++i) {
			may_recv_buf[i] = 1;
		}
		
		elapsed = get_time_sec() - start;
			
		int result = result_buf[source_rank];
		int response_type = elapsed >= limit ? ABORT : -1;
			
		writef("Got %i from %i\n", result, source_rank);

		// Check to see if we've received anything
		// from a producer
		if (source_rank >= (size / 2)) {
			_Assert(RANDV_MIN <= result && result < RANDV_MAX);
					
			if (job_q_len < size) {
				enqueue_buffer(job_q, size, job_q_len, result);
				job_q_len++;
				
				send_int_nb(&RESPONSE_ACK, source_rank);
			} else {
				if (response_type == ABORT) {
					send_int_nb(&response_type, source_rank);
					aborts_sent++; 
				} else {
					//_Assert(overflow_q_len < size);

					if (overflow_q_len < size) {
						enqueue_buffer(overflow_q, size, overflow_q_len, result);
						overflow_q_len++;
					}

					needs_ack[source_rank] = 1;
				}
			}
		// No, so we must have received something from a consumer
		} else if (0 < source_rank && source_rank < (size / 2)) {
			_Assert(result == REQ_WORK);

			if (response_type == ABORT) { // end state
				send_int_nb(&response_type,
										source_rank);
				aborts_sent++;
				consumer_aborts_sent++;
			} else if (job_q_len > 0) { // most frequent
				int v = dequeue_buffer(job_q, size);
				send_int_nb(&v,
										source_rank);
				job_q_len--;

				// handle any outstanding acks
				for (int i = 1; i < size; ++i) {
					if (needs_ack[i]) {
						send_int_nb(&RESPONSE_ACK, i);
						needs_ack[i] = 0;
					}
				}
			} else { // no work available, toggle off for the next iteration
				int v = NO_WORK;
				send_int_nb(&v, source_rank);

				may_recv_buf[source_rank] = 0;
			}
		}

		// we won't hit outstanding acks if
		// all the consumers are gone,
		// which means there won't be anymore aborts,
		// so we ensure that in this event that
		// we clean things up...
		if (consumer_aborts_sent == num_consumers) {
			for (int i = 1; i < size; ++i) {
				if (needs_ack[i]) {
					send_int_nb(&RESPONSE_ACK, i);
					needs_ack[i] = 0;
				}
			}
		}
						
		writef("aborts_sent: %i, elapsed: %li\n", aborts_sent, elapsed);
	}

	// Free main heap memory
	{
		free(request_buf);
		free(result_buf);
		free(may_recv_buf);
		free(needs_ack);
		free(job_q);
		free(overflow_q);
	}
	
	// Handle count report
	{
		int dummy = 0;
		int* consumed_counts = calloc(size, sizeof(*consumed_counts));
		_Assert(consumed_counts != NULL);
		
		MPICALL(MPI_Gather(&dummy,
											 1,
											 MPI_INT,
											 consumed_counts,
											 1,
											 MPI_INT,
											 0,
											 MPI_COMM_WORLD));

		int total = 0;
		for (int i = 0; i < size; ++i) {
			total += consumed_counts[i];
		}
		
		free(consumed_counts);

		printf("Total number of messages consumed: %i\n", total);
	}
}

void producer(int size, int rank) {
	_Assert(rank >= (size / 2));

	int iterate = 1;
	
	while (iterate) {
		int v = randv();

		{
			MPI_Request req = MPI_REQUEST_NULL;

			MPICALL(MPI_Isend(&v,
												1,
												MPI_INT,
												0,
												TAG,
												MPI_COMM_WORLD,
												&req));

			MPICALL(MPI_Wait(&req,
											 MPI_STATUS_IGNORE));
		}
		
		{
			int ack;
			MPI_Request req = MPI_REQUEST_NULL;
			
			MPICALL(MPI_Irecv(&ack,
												1,
												MPI_INT,
												0,
												TAG,
												MPI_COMM_WORLD,
												&req));
			
			MPICALL(MPI_Wait(&req, MPI_STATUS_IGNORE));

			_Assert(req == MPI_REQUEST_NULL);
			
			if (ack == ACK) {
				writef("[%i] ack\n", rank);
			} else if (ack == ABORT) {
				writef("[%i] abort\n", rank);
				iterate = 0;
			} else {
				writef("[%i] unknown %i\n", rank, ack);
				_Assert(0);
			}
		}
	}

	{
		int dummy = 0;

		MPICALL(MPI_Gather(&dummy,
											 1,
											 MPI_INT,
											 NULL,
											 1,
											 MPI_INT,
											 0,
											 MPI_COMM_WORLD));
	}
}

void consumer(int size, int rank) {
	_Assert(0 < rank && rank < (size / 2));

	int consume_count = 0;
	int iterate = 1;
	int QUERY = REQ_WORK;
	
	while (iterate) {
		int result = 0;
		
		{
			MPI_Request req = MPI_REQUEST_NULL;
			
			MPICALL(MPI_Isend(&QUERY,
												1,
												MPI_INT,
												0,
												TAG,
												MPI_COMM_WORLD,
												&req));

			MPICALL(MPI_Wait(&req, MPI_STATUS_IGNORE));
		}
		
		{
			MPI_Request req = MPI_REQUEST_NULL;
			
			MPICALL(MPI_Irecv(&result,
												1,
												MPI_INT,
												0,
												TAG,
												MPI_COMM_WORLD,
												&req));
			
			MPICALL(MPI_Wait(&req, MPI_STATUS_IGNORE));

			_Assert(req == MPI_REQUEST_NULL);
		}
		
		writef("[%i] consume_count: %i. result: %i\n",
					 rank,
					 consume_count,
					 result);

		if (result == ABORT) {
			iterate = 0;
		} else if (result != NO_WORK) {
			consume_count++;
		} 
	}

	writef("Final consume count: %i\n", consume_count);
	
	MPICALL(MPI_Gather(&consume_count,
										 1,
										 MPI_INT,
										 NULL,
										 1,
										 MPI_INT,
										 0,
										 MPI_COMM_WORLD));
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
	
	int rank, size;
	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	writef("size: %i", size);

	__rank = rank;
	
	_Assert(size == 4 ||
					size == 8 ||
					size == 12 ||
					size == 16);

	time_t limit = strtol(argv[1], NULL, 10);
	//_Assert(limit == 1);

	if (rank == 0) {
		broker(size, rank, limit);
	} else if (rank >= (size / 2)) {
		producer(size, rank);
	} else {
		consumer(size, rank);
	}

	writef("%s", "hit barrier");
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Finalize();

	return 0;
}
