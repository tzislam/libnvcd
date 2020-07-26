HOOK_ROOT := $(NVCD_HOME)/hook
HOOK_SRC_C := $(wildcard $(HOOK_ROOT)/src/*.cu)
HOOK_H := $(wildcard $(HOOK_ROOT)/include/nvcd/*.h)
HOOK_OBJ_C := $(HOOK_SRC_C:.cu=.o)

HOOK_TOBJ := $(HOOK_OBJ_C) $(HOOK_OBJ_CU)

HOOK_OBJDIR := $(HOOK_ROOT)/obj
HOOK_SRCDIR := $(HOOK_ROOT)/src

HOOK_OBJ := $(subst $(HOOK_SRCDIR), $(HOOK_OBJDIR), $(HOOK_TOBJ))

HOOK_LIB := libnvcdhook.so

$(HOOK_LIB): objdep $(HOOK_OBJ) $(HOOK_H) $(LIB)
	$(CXX) $(LD_FLAGS) -L$(NVCD_HOME)/bin $(HOOK_OBJ) -lnvcd $(LIBS) -o $(NVCD_HOME)/bin/$(HOOK_LIB)

$(HOOK_ROOT)/obj/%.o: $(HOOK_ROOT)/src/%.cu
	$(NVCC) $(NVCC_FLAGS) --compiler-options "-fPIC" -c $< -o $@
