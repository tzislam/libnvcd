include Makefile.inc

##
# Library
##

CC_FLAGS := $(INCLUDE) $(CC_FLAGS)

SRC_C := $(wildcard src/*.c)
OBJ_C := $(SRC_C:.c=.o)

LD_FLAGS := -shared
TOBJ := $(OBJ_C) $(OBJ_CU)

OBJDIR := obj
SRCDIR := src

OBJ := $(subst $(SRCDIR), $(OBJDIR), $(TOBJ))

LIB := libnvcd.so

$(LIB): $(OBJ)
	$(CC) $(CC_FLAGS) $(LD_FLAGS) $(OBJ) $(LIBS) -o bin/$(LIB)

$(LIB).pre: $(PRE)

obj/%.o: src/%.c objdep 
	$(CC) $(CC_FLAGS) -c $< -o $@


include $(NVCD_HOME)/nvcdrun/Makefile

include $(NVCD_HOME)/nvcdinfo/Makefile

include $(NVCD_HOME)/hook/Makefile

# Housekeeping
objdep:
	mkdir -p obj
	mkdir -p nvcdrun/obj
	mkdir -p nvcdinfo/obj
	mkdir -p hook/obj
	mkdir -p bin

clean:
	rm -f bin/*
	rm -f obj/*.o
	rm -f *~
	rm -f include/*~
	rm -f src/*~
	rm -f nvcdrun/src/*~
	rm -f nvcdinfo/src/*~
	rm -f nvcdrun/include/*~	
	rm -f nvcdinfo/include/*~
	rm -rf nvcdrun/obj
	rm -rf nvcdinfo/obj
	rm -rf hook/obj
	rm -rf $(UTIL_GROUP_INFO_DIRECTORY)

compile: $(HOOK_LIB) $(TEST_BIN) $(UTIL_BIN)
recompile: clean compile


