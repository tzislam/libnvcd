EXECS=random_rank
MPICC?=mpicc
FLAGS:=-g -ggdb

all: ${EXECS}

tmpi_rank.o: tmpi_rank.c
	${MPICC} $(FLAGS) -c tmpi_rank.c

random_rank: tmpi_rank.o parallel_rank.c
	${MPICC} $(FLAGS) -o random_rank parallel_rank.c tmpi_rank.o

clean:
	rm -f ${EXECS} *.o
