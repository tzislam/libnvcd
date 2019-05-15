CC=nvcc

CUDA_HOME ?= /usr/local/cuda-9.2
CUDA_ARCH_SM ?= sm_37

CUPTI := $(CUDA_HOME)/extras/CUPTI

INCLUDE=-I$(CUPTI)/include -I$(CUDA_HOME)/include

CXX_FLAGS=-Wall -Werror -g -ggdb -std=c++11

CC_FLAGS=-std=c++11 --compiler-options "$(CXX_FLAGS)"

ARCH=-arch=$(CUDA_ARCH_SM)
#ARCH=-arch=sm_60

SRC_C := $(wildcard *.cpp)
OBJ_C := $(SRC_C:.cpp=.o)

#-lnvidia-ml

LIBS=-L/usr/lib/x86_64-linux-gnu -L$(CUPTI)/lib64 -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lcupti

SRC_CU := $(wildcard *.cu)
OBJ_CU := $(SRC_CU:.cu=.o)

TOBJ := $(OBJ_C) $(OBJ_CU)
OBJDIR := obj
SRCDIR :=

FLAGS :=

TARGET := perfmon

$(TARGET): obj/gpumon.o obj/test.o obj/gpumon_cuda.o obj/util.o
	$(CC) $(FLAGS) $(LIBS) $(ARCH) $^ -o bin/$(TARGET)

obj/%.o: %.cpp objdep 
	g++ $(FLAGS) $(INCLUDE) $(CXX_FLAGS) -c $< -o $@

obj/%.o: %.cu objdep
	$(CC) $(FLAGS) $(INCLUDE) $(CC_FLAGS) $(ARCH) -c $< -o $@

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

