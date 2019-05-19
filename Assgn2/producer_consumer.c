#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

typedef struct qnode {
	qnode_t* next;
	int value;
} qnode_t;

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

void qbuf_free(qbuf_t* q) {
	assert(q q
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

	
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	
	
	MPI_Finalize();

	return 0;
}
