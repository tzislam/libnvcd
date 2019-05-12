CC=nvcc

INCLUDE=-I./include -I/usr/local/cuda-9.2/extras/CUPTI/include

CC_FLAGS=--compiler-options "-Wall -Werror -g -ggdb -std=c99"

ARCH=-arch=sm_37
#ARCH=-arch=sm_60

CUDA_HOME ?= /usr/local/cuda-9.2

SRC_C := $(wildcard src/*.c)
OBJ_C := $(SRC_C:.c=.o)

#-lnvidia-ml

LIBS=-L/usr/lib/x86_64-linux-gnu -L$(CUDA_HOME)/extras/CUPTI/lib64 -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lcupti

SRC_CU := $(wildcard src/*.cu)
OBJ_CU := $(SRC_CU:.cu=.o)

TOBJ := $(OBJ_C) $(OBJ_CU)
OBJDIR := obj
SRCDIR := src

FLAGS :=

OBJ := $(subst $(SRCDIR), $(OBJDIR), $(TOBJ))

TARGET := perfmon

$(TARGET): $(OBJ)
	$(CC) $(FLAGS) $(LIBS) $(ARCH) $(OBJ) -o bin/$(TARGET)

obj/%.o: src/%.c objdep 
	$(CC) $(FLAGS) $(INCLUDE) $(CC_FLAGS) $(ARCH) -c $< -o $@

obj/%.o: src/%.cu objdep
	$(CC) $(FLAGS) $(INCLUDE) $(ARCH) -c $< -o $@

objdep:
	mkdir -p obj
	mkdir -p bin

clean:
	rm -f bin/perfmon
	rm -f obj/*.o
	rm -f *~
	rm -f include/*~
	rm -f src/*~

#$$CUDACC -v $DEBUG -c $INCLUDE $ARCH src/gpu.cu -o obj/gpu.o &&\
#$CC -v $DEBUG $INCLUDE $ARCH -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcuda -lcudart obj/gpu.o src/main.c -o bin/perfmon

#gcc-7 -v -lnvidia-ml -lcuda -lcudart src/main.c -o bin/perfmon

