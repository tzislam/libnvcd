#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

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

void broker(int size, int rank) {
	assert(rank == 0);

	qbuf_t* job_q = qbuf_make(size);
	qbuf_t* o_job_q = qbuf_make(size);

	
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	
	
	MPI_Finalize();

	return 0;
}
