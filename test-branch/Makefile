INCLUDE=-I. -I../include -I$(CUPTI_INCLUDE)

CU_FLAGS=-std=c++11 --compiler-options "-Wall -lpthread -Werror -g -ggdb"

LD_FLAGS=-L/usr/lib/x86_64-linux-gnu -L./../bin -L$(CUDA_HOME)/lib64 -L$(CUPTI_LIB)

ARCH=-arch=$(CUDA_ARCH_SM)

LIBS=-lcuda -lcudart -lcurand -lcupti -lnvcd

FLAGS :=-G --compiler-options "-Wno-unused-function -Wno-unused-variable"

CC=nvcc

TARGET=test-branch

$(TARGET): test-branch.o
	$(CC) $(FLAGS) $(LD_FLAGS) $(LIBS) $(ARCH) $< -o $@

test-branch.o: test-branch.cu
	$(CC) $(FLAGS) $(INCLUDE) $(CU_FLAGS) $(ARCH) -c $^ -o $@
	$(CC) --ptx $(FLAGS) $(INCLUDE) $(CU_FLAGS) $(ARCH) -c $^ -o $@.ptx

clean:
	rm -f $(TARGET)
	rm -f *~
	rm -f *.o
	rm -f *.o.ptx

