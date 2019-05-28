CC=nvcc

CUDA_HOME ?= /usr/local/cuda-9.2
CUDA_ARCH_SM ?= sm_37

CUPTI := $(CUDA_HOME)/extras/CUPTI

##
# Library
##

INCLUDE=-I./include -I$(CUPTI)/include

CC_FLAGS=--compiler-options "-Wall -lpthread -Werror -g -ggdb -std=gnu99"

CU_FLAGS=-std=c++11 --compiler-options "-Wall -lpthread -Werror -g -ggdb"

LD_FLAGS=--compiler-options "-shared"

ARCH=-arch=$(CUDA_ARCH_SM)
#ARCH=-arch=sm_60

SRC_C := $(wildcard src/*.c)
OBJ_C := $(SRC_C:.c=.o)
PRE_C := $(SRC_C:.c=.c.pre)

LIBS=-L/usr/lib/x86_64-linux-gnu -L$(CUPTI)/lib64 -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lcupti

SRC_CU := $(wildcard src/*.cu)
OBJ_CU := $(SRC_CU:.cu=.o)
PRE_CU := $(SRC_CU:.cu=.cu.pre)

TOBJ := $(OBJ_C) $(OBJ_CU)

TPRE := $(PRE_C) $(PRE_CU)

OBJDIR := obj
SRCDIR := src
PREDIR := pre

FLAGS :=-G --compiler-options "-fPIC -fvisibility=hidden -Wno-unused-function -Wno-unused-variable"

FLAGS_PRE := --compiler-options "-E"

OBJ := $(subst $(SRCDIR), $(OBJDIR), $(TOBJ))
PRE := $(subst $(SRCDIR), $(PREDIR), $(TPRE))

LIB := libnvcd.so


##
# Test binary
##

TEST_SRC_C := $(wildcard test/src/*.cpp)
TEST_OBJ_C := $(TEST_SRC_C:.cpp=.o)
TEST_PRE_C := $(TEST_SRC_C:.cpp=.cpp.pre)

TEST_SRC_CU := $(wildcard test/src/*.cu)
TEST_OBJ_CU := $(TEST_SRC_CU:.cu=.o)
TEST_PRE_CU := $(TEST_SRC_CU:.cu=.cu.pre)

TEST_OBJDIR := test/obj
TEST_SRCDIR := test/src
TEST_PREDIR := test/pre

TEST_OBJ := $(subst $(TEST_SRCDIR), $(TEST_OBJDIR), $(TEST_OBJ_C) $(TEST_OBJ_CU))
TEST_PRE := $(subst $(TEST_SRCDIR), $(TEST_PREDIR), $(TEST_PRE_C) $(TEST_PRE_CU))

TEST_BIN := nvcdrun

TEST_CC_FLAGS=-G -std=c++11 --compiler-options "-std=c++11 -Wall -lpthread -Werror -g -ggdb -Wno-unused-function -Wno-unused-variable"

TEST_LIBS := $(LIBS) -Lbin -lnvcd

TEST_INCLUDE := $(INCLUDE) -I./test/include

TEST_FLAGS_PRE := $(FLAGS_PRE)

##
# Targets
##

# Lib
$(LIB): $(OBJ)
	$(CC) $(FLAGS) $(LD_FLAGS) $(LIBS) $(ARCH) $(OBJ) -o bin/$(LIB)

$(LIB).pre: $(PRE)

obj/%.o: src/%.c objdep 
	$(CC) $(FLAGS) $(INCLUDE) $(CC_FLAGS) $(ARCH) -c $< -o $@

obj/%.o: src/%.cu objdep
	$(CC) $(FLAGS) $(INCLUDE) $(CU_FLAGS) $(ARCH) -c $< -o $@
	$(CC) --ptx $(FLAGS) $(INCLUDE) $(CU_FLAGS) $(ARCH) -c $< -o $@.ptx

pre/%.c.pre: src/%.c objdep
	$(CC) $(FLAGS) $(FLAGS_PRE) $(INCLUDE) $(CC_FLAGS) $(ARCH) -c $< -o $@

pre/%.cu.pre: src/%.cu objdep
	$(CC) $(FLAGS) $(FLAGS_PRE) $(INCLUDE) $(CU_FLAGS) $(ARCH) -c $< -o $@

# Test binary
$(TEST_BIN): $(TEST_OBJ) $(LIB)
	$(CC) $(TEST_INCLUDE) $(TEST_CC_FLAGS) $(TEST_LIBS) $(ARCH) $(TEST_OBJ) -o bin/$(TEST_BIN)

$(TEST_BIN).pre: $(TEST_PRE) $(LIB).pre

test/obj/%.o: test/src/%.cpp objdep
	$(CC) $(TEST_INCLUDE) $(TEST_CC_FLAGS) $(ARCH) -c $< -o $@

test/obj/%.o: test/src/%.cu objdep
	$(CC) $(TEST_INCLUDE) $(TEST_CC_FLAGS) $(ARCH) -c $< -o $@
	$(CC) --ptx $(TEST_INCLUDE) $(TEST_CC_FLAGS) $(ARCH) -c $< -o $@.ptx

test/pre/%.c.pre: test/src/%.c objdep
	$(CC) $(TEST_FLAGS_PRE) $(TEST_INCLUDE) $(TEST_CC_FLAGS) $(ARCH) -c $< -o $@

test/pre/%.cu.pre: test/src/%.cu objdep
	$(CC) $(TEST_FLAGS_PRE) $(TEST_INCLUDE) $(TEST_CC_FLAGS) $(ARCH) -c $< -o $@

# Housekeeping
objdep:
	mkdir -p obj
	mkdir -p bin
	mkdir -p test/obj
	mkdir -p pre
	mkdir -p test/pre

clean:
	rm -f bin/perfmon
	rm -f obj/*.o
	rm -f test/obj/*.o
	rm -f test/src/*~
	rm -f test/pre/*
	rm -f pre/*
	rm -f *~
	rm -f include/*~
	rm -f src/*~

#$$CUDACC -v $DEBUG -c $INCLUDE $ARCH src/gpu.cu -o obj/gpu.o &&\
#$CC -v $DEBUG $INCLUDE $ARCH -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcuda -lcudart obj/gpu.o src/main.c -o bin/perfmon

#gcc-7 -v -lnvidia-ml -lcuda -lcudart src/main.c -o bin/perfmon

