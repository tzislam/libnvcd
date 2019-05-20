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

typedef struct qnode qnode_t;

struct qnode {
	qnode_t* next;
	int value;
	int rank;
};

typedef struct qbuf {
	qnode_t* head;
	qnode_t* tail;

	int count;
	int capacity;
} qbuf_t;

qnode_t* qnode_make(int v, int r) {
	qnode_t* qn = malloc(sizeof(qnode_t));
	_Assert(qn != NULL);

	qn->value = v;
	qn->rank = r;
	qn->next = NULL;
	
	return qn;
}

void qbuf_free(qbuf_t** pq) {
	qbuf_t* q = *pq;
	_Assert(q != NULL);

	qnode_t* h = q->head;
	
	while (h != NULL) {
		if (h->next == NULL) {
			_Assert(h == q->tail);
		}

		qnode_t* tmp = h->next;

		free(h);

		h = tmp;
	}

	free(q);
	*pq = NULL;
}

qnode_t* qbuf_dequeue(qbuf_t* q) {
	_Assert(q != NULL);
	_Assert(q->head != NULL
				 && q->tail != NULL);
	_Assert(q->count > 0);

	qnode_t* n = q->head;
	q->head = q->head->next;

	if (q->head == NULL) {
		q->tail = NULL;
	}
	
	n->next = NULL;
	
	q->count--;
	
	return n;
}

void qbuf_enqueue(qbuf_t* q, int v, int r) {
	_Assert(q != NULL);
	_Assert(q->count < q->capacity);
	
	qnode_t* t = qnode_make(v, r);
	
	if (q->head == NULL || q->tail == NULL) {
		_Assert(q->head == NULL && q->tail == NULL);

		q->head = q->tail = t;
	} else {
		q->tail->next = t;
		q->tail = t;
	}

	q->count++;
}

qbuf_t* qbuf_make(int size) {
	qbuf_t* q = malloc(sizeof(qbuf_t));
	_Assert(q != NULL);

	q->head = NULL;
	q->tail = NULL;
	
	q->capacity = size;
	q->count = 0;
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

#define WRITEFBUFLEN 4096

static char __writef_buffer[WRITEFBUFLEN + 1024] = { 0 };
static int __writef_counter = 0;

void __writef(int rank, const char* func, int line, char* fmt, ...) {
	{
		int64_t micro = get_time();
		int k = sprintf(&__writef_buffer[__writef_counter], "[%" PRId64 "]: %s %i|", micro, func, line);

		__writef_counter += k;
		
		int l = 0;
		{
			va_list arg;
			va_start(arg, fmt);
			l = vsprintf(&__writef_buffer[__writef_counter], fmt, arg);
			va_end(arg);
		}

		__writef_counter += l;
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

	qbuf_t* job_q = qbuf_make(size);
	qbuf_t* o_job_q = qbuf_make(size);

	time_t start = get_time_sec();
	time_t elapsed = get_time_sec() - start;

	int RESPONSE_ACK = ACK;

	// incremented each time when an abort is sent.
	// once the amount is equal to size,
	// the loop is finished.
	int aborts_sent = 1;

	MPI_Request* request_buf = malloc(sizeof(*request_buf) * size);
	int* result_buf = malloc(sizeof(*result_buf) * size);
	int* may_recv_buf = malloc(sizeof(*may_recv_buf) * size);
	
	_Assert(request_buf != NULL);
	_Assert(result_buf != NULL);
	_Assert(may_recv_buf != NULL);
 
	for (int i = 1; i < size; ++i) {
		request_buf[i] = MPI_REQUEST_NULL;
		result_buf[i] = -1;
		may_recv_buf[i] = 1;
	}
	
	while (aborts_sent < size) {
		writef("%s", "iteration start");

		volatile int k = 0;
		// handle outstanding acks from outstanding queue
		while (job_q->count < job_q->capacity
					 && o_job_q->count > 0) {
			
			qnode_t* n = qbuf_dequeue(o_job_q);
			qbuf_enqueue(job_q, n->value, n->rank);

			writef("[%i] outstanding for (value, rank) = (%i, %i): sending ack\n",
						 k,
						 n->value,
						 n->rank);
			
			send_int_nb(&RESPONSE_ACK, n->rank);

			free(n);
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

		// ensure that all flags are toggled on
		// for next iteration (unless the next
		// messaged received deems that one flag
		// must be turned off)
		for (int i = 1; i < size; ++i) {
			may_recv_buf[i] = 1;
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
			
		elapsed = get_time_sec() - start;
			
		int result = result_buf[source_rank];
		int response_type = elapsed >= limit ? ABORT : -1;
			
		writef("Got %i from %i\n", result, source_rank);

		// Check to see if we've received anything
		// from a producer
		if (source_rank >= (size / 2)) {
			_Assert(RANDV_MIN <= result && result < RANDV_MAX);
					
			if (job_q->count < job_q->capacity) {
				qbuf_enqueue(job_q, result, source_rank);
				send_int_nb(&RESPONSE_ACK, source_rank);
			} else {
				if (response_type == ABORT) {
					send_int_nb(&response_type, source_rank);
					aborts_sent++; 
				} else {
					qbuf_enqueue(o_job_q, result, source_rank);
				}
			}

		// So we must have received something from a consumer
		} else if (0 < source_rank && source_rank < (size / 2)) {
			_Assert(result == REQ_WORK);

			if (response_type == ABORT) { // end state
				send_int_nb(&response_type, source_rank);
				aborts_sent++;
			} else if (job_q->count > 0) { // most frequent
				qnode_t* n = qbuf_dequeue(job_q);
				send_int_nb(&n->value, source_rank);
				free(n);
			} else { // no work available, toggle off for the next iteration
				int v = NO_WORK;
				send_int_nb(&v, source_rank);

				for (int i = 1; i < (size / 2); ++i) {
					may_recv_buf[i] = 0;
				} 
			}
		}
						
		writef("aborts_sent: %i, elapsed: %li\n", aborts_sent, elapsed);
	}
}

void producer(int size, int rank) {
	_Assert(rank >= (size / 2));

	int iterate = 1;
	
	while (iterate) {
		int v = randv();

		MPI_Request req;
		MPICALL(MPI_Isend(&v,
											1,
											MPI_INT,
											0,
											TAG,
											MPI_COMM_WORLD,
											&req));

		MPICALL(MPI_Wait(&req,
										 MPI_STATUS_IGNORE));
		
		{
			int ack;
			
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
}

void consumer(int size, int rank) {
	_Assert(0 < rank && rank < (size / 2));

	int consume_count = 0;
	int iterate = 1;
	int QUERY = REQ_WORK;
	
	while (iterate) {
		MPI_Request req = MPI_REQUEST_NULL;
		MPI_Status stat = { 0 };
		int result = 0;
		
		{
			MPICALL(MPI_Isend(&QUERY,
												1,
												MPI_INT,
												0,
												TAG,
												MPI_COMM_WORLD,
												&req));

			MPICALL(MPI_Wait(&req, &stat));
		}
		
		{
			MPICALL(MPI_Irecv(&result,
												1,
												MPI_INT,
												0,
												TAG,
												MPI_COMM_WORLD,
												&req));
			
			MPICALL(MPI_Wait(&req, &stat));

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
