CC := gcc

INCLUDE := -I./src

FLAGS := -std=gnu99 -g -ggdb 

SRC_C := $(wildcard src/*.c)
OBJ_C := $(SRC_C:.c=.o)

LIBS := -fopenmp

TOBJ := $(OBJ_C)

OBJDIR := obj
SRCDIR := src

OBJ := $(subst $(SRCDIR), $(OBJDIR), $(TOBJ))

TARGET := affinity_example

affinity_example: src/affinity_example.c bingen
	$(CC) $(FLAGS) $(LIBS) $< -o bin/$@

bingen:
	mkdir -p bin

objgen:
	mkdir -p obj

dirgen: bingen objgen

clean:
	rm -f *~
	rm -f src/*~
	rm -f bin/*
	rm -f obj/*.o
