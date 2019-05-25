CC=nvcc

CUDA_HOME ?= /usr/local/cuda-9.2
CUDA_ARCH_SM ?= sm_37

CUPTI := $(CUDA_HOME)/extras/CUPTI

##
# Library
##

INCLUDE=-I./include -I$(CUPTI)/include

CC_FLAGS=--compiler-options "-Wall -lpthread -Werror -g -ggdb -std=gnu99"

CU_FLAGS=-std=c++11 -G

LD_FLAGS=--compiler-options "-shared"

ARCH=-arch=$(CUDA_ARCH_SM)
#ARCH=-arch=sm_60

SRC_C := $(wildcard src/*.c)
OBJ_C := $(SRC_C:.c=.o)

LIBS=-L/usr/lib/x86_64-linux-gnu -L$(CUPTI)/lib64 -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lcupti

SRC_CU := $(wildcard src/*.cu)
OBJ_CU := $(SRC_CU:.cu=.o)

TOBJ := $(OBJ_C) $(OBJ_CU)
OBJDIR := obj
SRCDIR := src

FLAGS := --compiler-options "-fPIC -fvisibility=hidden -Wno-unused-function"

OBJ := $(subst $(SRCDIR), $(OBJDIR), $(TOBJ))

LIB := libnvcd.so


##
# Test binary
##

TEST_SRC_C := $(wildcard test/src/*.c)
TEST_OBJ_C := $(TEST_SRC_C:.c=.o)

TEST_SRC_CU := $(wildcard test/src/*.cu)
TEST_OBJ_CU := $(TEST_SRC_CU:.cu=.o)

TEST_OBJDIR := test/obj
TEST_SRCDIR := test/src

TEST_OBJ := $(subst $(TEST_SRCDIR), $(TEST_OBJDIR), $(TEST_OBJ_C) $(TEST_OBJ_CU))

TEST_BIN := nvcdrun

TEST_CC_FLAGS=-std=c++11 --compiler-options "-std=c++11 -Wall -lpthread -Werror -g -ggdb"

TEST_LIBS := $(LIBS) -Lbin -lnvcd

TEST_INCLUDE := $(INCLUDE) -I./test/include

##
# Targets
##

# Lib
$(LIB): $(OBJ)
	$(CC) $(FLAGS) $(LD_FLAGS) $(LIBS) $(ARCH) $(OBJ) -o bin/$(LIB)

obj/%.o: src/%.c objdep 
	$(CC) $(FLAGS) $(INCLUDE) $(CC_FLAGS) $(ARCH) -c $< -o $@

obj/%.o: src/%.cu objdep
	$(CC) $(FLAGS) $(INCLUDE) $(CU_FLAGS) $(ARCH) -c $< -o $@
	$(CC) --ptx $(FLAGS) $(INCLUDE) $(CU_FLAGS) $(ARCH) -c $< -o $@.ptx

# Test binary
$(TEST_BIN): $(TEST_OBJ) $(LIB)
	$(CC) $(TEST_INCLUDE) $(TEST_CC_FLAGS) $(TEST_LIBS) $(ARCH) $(TEST_OBJ) -o bin/$(TEST_BIN)

test/obj/%.o: test/src/%.c objdep
	$(CC) $(TEST_INCLUDE) $(TEST_CC_FLAGS) $(ARCH) -c $< -o $@

test/obj/%.o: test/src/%.cu objdep
	$(CC) $(TEST_INCLUDE) $(TEST_CC_FLAGS) $(ARCH) -c $< -o $@
	$(CC) --ptx $(TEST_INCLUDE) $(TEST_CC_FLAGS) $(ARCH) -c $< -o $@.ptx

# Housekeeping
objdep:
	mkdir -p obj
	mkdir -p bin
	mkdir -p test/obj

clean:
	rm -f bin/perfmon
	rm -f obj/*.o
	rm -f test/obj/*.o
	rm -f test/src/*~
	rm -f *~
	rm -f include/*~
	rm -f src/*~

#$$CUDACC -v $DEBUG -c $INCLUDE $ARCH src/gpu.cu -o obj/gpu.o &&\
#$CC -v $DEBUG $INCLUDE $ARCH -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcuda -lcudart obj/gpu.o src/main.c -o bin/perfmon

#gcc-7 -v -lnvidia-ml -lcuda -lcudart src/main.c -o bin/perfmon

