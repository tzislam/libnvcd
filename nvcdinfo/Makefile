UTIL_ROOT := $(NVCD_HOME)/nvcdinfo

UTIL_SRC_C := $(wildcard $(UTIL_ROOT)/src/*.cpp)
UTIL_OBJ_C := $(UTIL_SRC_C:.cpp=.o)

UTIL_SRC_CU := $(wildcard $(UTIL_ROOT)/src/*.cu)
UTIL_OBJ_CU := $(UTIL_SRC_CU:.cu=.co)

UTIL_OBJDIR := $(UTIL_ROOT)/obj
UTIL_SRCDIR := $(UTIL_ROOT)/src

UTIL_OBJ := $(subst $(UTIL_SRCDIR), $(UTIL_OBJDIR), $(UTIL_OBJ_C) $(UTIL_OBJ_CU))

UTIL_BIN := nvcdinfo

UTIL_CXX_FLAGS=$(CXX_FLAGS) -I$(UTIL_ROOT)/include
UTIL_NVCC_FLAGS=$(NVCC_FLAGS) -I$(UTIL_ROOT)/include

UTIL_LIBS := $(LIBS) -L$(NVCD_HOME)/bin -lnvcd

$(UTIL_BIN): $(UTIL_OBJ) $(LIB)
	$(CXX) $(UTIL_CXX_FLAGS) $(UTIL_LIBS) $(UTIL_OBJ) -o $(NVCD_HOME)/bin/$(UTIL_BIN)

$(UTIL_ROOT)/obj/%.o: $(UTIL_ROOT)/src/%.cpp objdep
	$(CXX) $(UTIL_CXX_FLAGS) $(ARCH) -c $< -o $@

$(UTIL_ROOT)/obj/%.co: $(UTIL_ROOT)/src/%.cu objdep
	$(NVCC) $(UTIL_NVCC_FLAGS) $(NVCC_ARCH) -c $< -o $@
	$(NVCC) --ptx $(UTIL_NVCC_FLAGS) $(NVCC_ARCH) -c $< -o $@.ptx
