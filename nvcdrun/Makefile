TEST_ROOT := $(NVCD_HOME)/nvcdrun

TEST_SRC_C := $(wildcard $(TEST_ROOT)/src/*.c)
TEST_OBJ_C := $(TEST_SRC_C:.c=.o)

TEST_SRC_CU := $(wildcard $(TEST_ROOT)/src/*.cu)
TEST_OBJ_CU := $(TEST_SRC_CU:.cu=.co)

TEST_OBJDIR := $(TEST_ROOT)/obj
TEST_SRCDIR := $(TEST_ROOT)/src

TEST_OBJ := $(subst $(TEST_SRCDIR), $(TEST_OBJDIR), $(TEST_OBJ_C) $(TEST_OBJ_CU))

TEST_LIBS := $(LIBS) -L$(NVCD_HOME)/bin -lnvcd

TEST_BIN := nvcdrun

TEST_CXX_FLAGS := $(CXX_FLAGS) -I$(TEST_ROOT)/include

TEST_NVCC_FLAGS := $(NVCC_FLAGS) -I$(TEST_ROOT)/include

$(TEST_BIN): $(TEST_OBJ) $(LIB)
	$(CXX) $(TEST_CXX_FLAGS) $(TEST_LIBS) $(TEST_OBJ) -o $(NVCD_HOME)/bin/$(TEST_BIN)

$(TEST_BIN).pre: $(TEST_PRE) $(LIB).pre

$(TEST_ROOT)/obj/%.o: $(TEST_ROOT)/src/%.c objdep
	$(CXX) $(TEST_CXX_FLAGS) -c $< -o $@

$(TEST_ROOT)/obj/%.co: $(TEST_ROOT)/src/%.cu objdep
	$(NVCC) $(TEST_NVCC_FLAGS) $(NVCC_ARCH) -c $< -o $@
	$(NVCC) --ptx  $(TEST_NVCC_FLAGS) $(NVCC_ARCH) -c $< -o $@.ptx





