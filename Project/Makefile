CC=nvcc

INCLUDE=-I./include -I/usr/local/cuda-9.2/extras/CUPTI/include

CUDA_DBG=--compiler-options "-rdynamic -g -ggdb -std=c++0x" 
DEBUG=-G $(CUDA_DBG)

ARCH=-arch=sm_50
#ARCH=-arch=sm_60


SRC_CPP := $(wildcard src/*.cpp)
OBJ_CPP := $(SRC_CPP:.cpp=.o)

#-lnvidia-ml

LIBS=-L/usr/lib/x86_64-linux-gnu -L/usr/local/cuda-9.2/extras/CUPTI/lib64 -lcuda -lcudart -lcupti

SRC_CU := $(wildcard src/*.cu)
OBJ_CU := $(SRC_CU:.cu=.o)

TOBJ := $(OBJ_CPP) $(OBJ_CU)
OBJDIR := obj
SRCDIR := src

FLAGS :=

OBJ := $(subst $(SRCDIR), $(OBJDIR), $(TOBJ))

TARGET := perfmon

$(TARGET): $(OBJ)
	$(CC) $(FLAGS) $(LIBS) $(DEBUG) $(ARCH) $(OBJ) -o bin/$(TARGET)

obj/%.o: src/%.cpp objdep 
	$(CC) $(FLAGS) $(INCLUDE) $(DEBUG) $(ARCH) -c $< -o $@

obj/%.o: src/%.cu objdep
	$(CC) $(FLAGS) $(INCLUDE) $(DEBUG) $(ARCH) -c $< -o $@

objdep:
	mkdir -p obj

clean:
	rm -f bin/perfmon
	rm -f obj/*.o
	rm -f *~
	rm -f include/*~
	rm -f src/*~

#$$CUDACC -v $DEBUG -c $INCLUDE $ARCH src/gpu.cu -o obj/gpu.o &&\
#$CC -v $DEBUG $INCLUDE $ARCH -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcuda -lcudart obj/gpu.o src/main.cpp -o bin/perfmon

#gcc-7 -v -lnvidia-ml -lcuda -lcudart src/main.cpp -o bin/perfmon

