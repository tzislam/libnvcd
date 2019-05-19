# Sample makefile 
CC=mpicc

all: producer_consumer workpool
pc: producer_consumer.c
	${CC} -o producer_consumer producer_consumer.c
wp: workpool.c
	${CC} -o workpool workpool.c

clean:
	rm -f compute_stddev
