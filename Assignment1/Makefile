CC := gcc

INCLUDE := -I./src

FLAGS := -std=gnu99 -g -ggdb 

SRC_C := $(wildcard src/*.c)
OBJ_C := $(SRC_C:.c=.o)

LIBS= -fopenmp

TOBJ := $(OBJ_C)

OBJDIR := obj
SRCDIR := src

OBJ := $(subst $(SRCDIR), $(OBJDIR), $(TOBJ))

TARGET := test

bin/$(TARGET): $(OBJ)
	$(CC) $(FLAGS) $(LIBS) $(OBJ) -o $@

obj/%.o: src/%.c dirgen
	$(CC) $(FLAGS) $(INCLUDE) $(DEBUG) $(ARCH) -c $< -o $@

dirgen:
	mkdir -p bin
	mkdir -p obj

clean:
	rm -f *~
	rm -f bin/perfmon
	rm -f obj/*.o
