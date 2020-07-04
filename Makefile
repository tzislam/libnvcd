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
	$(CC) $(CC_FLAGS) $(LD_FLAGS) $(LIBS) $(OBJ) -o bin/$(LIB)

$(LIB).pre: $(PRE)

obj/%.o: src/%.c objdep 
	$(CC) $(CC_FLAGS) -c $< -o $@


include $(NVCD_HOME)/nvcdrun/Makefile

include $(NVCD_HOME)/nvcdinfo/Makefile

# Housekeeping
objdep:
	mkdir -p obj
	mkdir -p nvcdrun/obj
	mkdir -p nvcdinfo/obj
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

#$$CUDACC -v $DEBUG -c $INCLUDE $ARCH src/gpu.cu -o obj/gpu.o &&\
#$CC -v $DEBUG $INCLUDE $ARCH -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcuda -lcudart obj/gpu.o src/main.c -o bin/perfmon

#gcc-7 -v -lnvidia-ml -lcuda -lcudart src/main.c -o bin/perfmon

