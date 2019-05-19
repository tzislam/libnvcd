#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>

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
	assert(qn != NULL);

	qn->value = v;
	qn->next = NULL;
	
	return qn;
}

void qbuf_free(qbuf_t** pq) {
	qbuf_t* q = *pq;
	assert(q != NULL);

	qnode_t* h = q->head;
	
	while (h != NULL) {
		if (h->next == NULL) {
			assert(h == q->tail);
		}

		qnode_t* tmp = h->next;

		free(h);

		h = tmp;
	}

	free(q);
	*pq = NULL;
}

int qbuf_deqeue(qbuf_t* q) {
	assert(q != NULL);
	assert(q->head != NULL
				 && q->tail != NULL);
	assert(q->count > 0);

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
	assert(q != NULL);
	assert(q->count < q->capacity);
	
	qnode_t* t = qnode_make(v);
	
	if (q->head == NULL || q->tail == NULL) {
		assert(q->head == NULL && q->tail == NULL);

		q->head = q->tail = t;
	} else {
		q->tail->next = t;
		q->tail = t;
	}

	q->count++;
}

qbuf_t* qbuf_make(int size) {
	qbuf_t* q = malloc(sizeof(qbuf_t));
	assert(q != NULL);

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
	assert(s->MPI_ERROR == MPI_SUCCESS);
}

void recv_int(int* value, int src, MPI_Status* status) {
	assert(value != NULL);
	assert(status != NULL);

	MPI_Recv(value,
					 1,
					 MPI_INT,
					 src,
					 MPI_ANY_TAG,
					 MPI_COMM_WORLD,
					 status);

	assert_status(status);
	assert(*value >= 0);
}


void assert_isend(MPI_Request req) {
	int flag = 0;
	MPI_Status s;
	MPI_Request_get_status(req, &flag, &s);
	assert(flag == 1 && s.MPI_ERROR == MPI_SUCCESS);
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
	
	assert_isend(req);

	return req;
}

void broker(int size, int rank, time_t limit) {
	assert(rank == 0);

	qbuf_t* job_q = qbuf_make(size);
	qbuf_t* o_job_q = qbuf_make(size);

	time_t start = get_time();
	time_t elapsed = get_time() - start;
	
	while (elapsed < limit) {
		int result = -1;

		MPI_Status s;
		recv_int(&result, MPI_ANY_SOURCE, &s);
		printf("value received: %i, from %i:\n", result, s.MPI_SOURCE);

		int response = ACK;
		
		elapsed = get_time() - start;
		if (elapsed >= limit) {
			response = ABORT;
		}
		printf("elapsed: %li\n", elapsed);

		send_int_nb(&response, s.MPI_SOURCE);
	}

	puts("Time's up");
}


void producer(int size, int rank) {
	assert(rank >= (size / 2));

	int iterate = 1;
	
	while (iterate) {
		int v = 233;
		send_int_nb(&v, 0);

		MPI_Status s;
		int ack;
		recv_int(&ack, 0, &s);
		assert(s.MPI_SOURCE == 0);

		if (ack == ACK) {
			printf("[%i] ack\n", rank);
		} else if (ack == ABORT) {
			printf("[%i] abort\n", rank);
			iterate = 0;
		} else {
			assert(0);
		}
	}
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int rank, size;
	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	assert(size == 4 || size == 8 || size == 12 || size == 16);

	time_t limit = strtol(argv[1], NULL, 10);
	assert(limit == 10);

	if (rank == 0) {
		broker(size, rank, limit);
	} else if (rank >= (size / 2)) {
		producer(size, rank);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Finalize();

	return 0;
}
