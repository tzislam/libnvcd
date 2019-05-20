# Sample makefile 
CC=mpicc

all: producer_consumer workpool
pc: producer_consumer.c

	${CC} -g -ggdb -O0 -o producer_consumer producer_consumer.c
wp: workpool.c
	${CC} -g -ggdb -O0 -o workpool workpool.c

clean:
	rm -f producer_consumer
	rm -f workpool
