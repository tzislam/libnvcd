#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>
#include <setjmp.h>
#include <execinfo.h>
#include <string.h>

#define ST_PRINT_BUF_SZ 4096

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
		stacktrace();
		longjmp(_jmp_buf, 1);
	}
}

#define _Assert(cond) assert_impl(cond, #cond, __FILE__, __LINE__)

typedef struct qnode qnode_t;

struct qnode {
	qnode_t* next;
	int value;
};

typedef struct qbuf {
	qnode_t* head;
	qnode_t* tail;

	int count;
	int capacity;
} qbuf_t;

qnode_t* qnode_make(int v) {
	qnode_t* qn = malloc(sizeof(qnode_t));
	_Assert(qn != NULL);

	qn->value = v;
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

int qbuf_deqeue(qbuf_t* q) {
	_Assert(q != NULL);
	_Assert(q->head != NULL
				 && q->tail != NULL);
	_Assert(q->count > 0);

	int ret = q->head->value;

	{
		qnode_t* n = q->head;
		q->head = q->head->next;
		free(n);
	}
	
	q->count--;
	
	return ret;
}

void qbuf_enqeue(qbuf_t* q, int v) {
	_Assert(q != NULL);
	_Assert(q->count < q->capacity);
	
	qnode_t* t = qnode_make(v);
	
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
	ABORT = 255,
	ACK = 42
};

#define TAG 123

time_t get_time() {
	return time(NULL);
}

void assert_status(MPI_Status *s) {
	_Assert(s->MPI_ERROR == MPI_SUCCESS);
}

void recv_int(int* value, int src, MPI_Status* status) {
	_Assert(value != NULL);
	_Assert(status != NULL);

	MPI_Recv(value,
					 1,
					 MPI_INT,
					 src,
					 MPI_ANY_TAG,
					 MPI_COMM_WORLD,
					 status);

	assert_status(status);
	_Assert(*value >= 0);
}


void assert_isend(MPI_Request req) {
	int flag = 0;
	MPI_Status s;
	MPI_Request_get_status(req, &flag, &s);
	
	_Assert(flag == 1);
	_Assert(s.MPI_ERROR == MPI_SUCCESS);
}

MPI_Request send_int_nb(int* value, int dest) {
	MPI_Request req;
	MPI_Isend(value,
						1,
						MPI_INT,
						0,
						TAG,
						MPI_COMM_WORLD,
						&req);
	
	//assert_isend(req);

	return req;
}

void broker(int size, int rank, time_t limit) {
	_Assert(rank == 0);

	qbuf_t* job_q = qbuf_make(size);
	qbuf_t* o_job_q = qbuf_make(size);

	time_t start = get_time();
	time_t elapsed = get_time() - start;
	
	while (elapsed < limit) {
		int result = -1;

		MPI_Status s;
		recv_int(&result, MPI_ANY_SOURCE, &s);
		if (s.MPI_SOURCE != 0) {
			printf("value received: %i, from %i:\n", result, s.MPI_SOURCE);

			int response = ACK;
		
			elapsed = get_time() - start;
			if (elapsed >= limit) {
				response = ABORT;
			}
			printf("elapsed: %li\n", elapsed);

			send_int_nb(&response, s.MPI_SOURCE);
		}
	}

	puts("Time's up");
}


void producer(int size, int rank) {
	_Assert(rank >= (size / 2));

	int iterate = 1;
	
	while (iterate) {
		int v = 233;
		MPI_Request req = send_int_nb(&v, 0);

		{
			MPI_Status s;
			MPI_Wait(&req, &s);
			_Assert(s
		}
		{
			MPI_Status s;
			int ack;
			recv_int(&ack, 0, &s);
			_Assert(s.MPI_SOURCE == 0);

			if (ack == ACK) {
				printf("[%i] ack\n", rank);
			} else if (ack == ABORT) {
				printf("[%i] abort\n", rank);
				iterate = 0;
			} else {
				_Assert(0);
			}
		}
	}
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int __jmp = setjmp(_jmp_buf);

	if (__jmp == 0) {	
		int rank, size;
	
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		_Assert(size == 4 || size == 8 || size == 12 || size == 16);

		time_t limit = strtol(argv[1], NULL, 10);
		_Assert(limit == 1);

		if (rank == 0) {
			broker(size, rank, limit);
		} else if (rank >= (size / 2)) {
			producer(size, rank);
		}
	}

	if (__jmp == 1) {
		puts("ASSERT FAILURE");
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Finalize();

	return 0;
}
