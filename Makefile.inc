###
# Global include variables
###

# https://stackoverflow.com/a/18137056
# these should work regardless of where this is included from
mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))

# get absolute path, with trailing slash removed
NVCD_HOME := $(patsubst %/,%,$(dir $(mkfile_path)))

CC=gcc
NVCC=nvcc
CXX=g++

CUDA_HOME ?= /usr/local/cuda-10.1
CUDA_ARCH_SM ?= sm_70

CUPTI := $(CUDA_HOME)/extras/CUPTI

CC_FLAGS :=
CU_FLAGS :=

BASE_FLAGS := -Wall -lpthread -Werror -Wno-unused-function -Wno-unused-variable

INCLUDE=-I$(NVCD_HOME)/include -I$(CUPTI)/include -I$(CUDA_HOME)/include

CC_SO_FLAGS := -fPIC -fvisibility=hidden

CC_STD := -std=gnu99
NVCC_STD := -std=c++11
CXX_STD := -std=c++11 

CC_FLAGS := $(CC_STD) $(INCLUDE) $(BASE_FLAGS) $(CC_SO_FLAGS)
CXX_FLAGS := $(CXX_STD) $(INCLUDE) $(BASE_FLAGS)
NVCC_FLAGS := $(NVCC_STD) $(INCLUDE) --compiler-options "$(BASE_FLAGS)"

LIBS=-L/usr/lib/x86_64-linux-gnu -L$(CUPTI)/lib64 -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lcupti -ldl

NVCC_ARCH := "-arch=$(CUDA_ARCH_SM)"
#
# These are used in $(UTIL_GROUP_INFO) target, in nvcdinfo/Makefile:
#
# this is the device ID parameter that is passed to nvcdinfo
NVCDINFO_DEVICE_ID ?= 0
#
# this is the maximum size of each group of events in the associated CSV
NVCDINFO_GROUP_SIZE ?= 5

ifeq ($(DEBUG),1)
	CC_FLAGS := $(CC_FLAGS) -g -ggdb -O0
	NVCC_FLAGS := $(NVCC_FLAGS) -O0 -G --compiler-options "-g -ggdb -O0"
	CXX_FLAGS := $(CXX_FLAGS) -g -ggdb -O0
else
	CC_FLAGS := $(CC_FLAGS) -O2
	NVCC_FLAGS := $(NVCC_FLAGS) -O2 --compiler-options "-O2"
	CXX_FLAGS := $(CXX_FLAGS) -O2
endif



$(info NVCC_FLAGS =  $(NVCC_FLAGS))
$(info NVCD_HOME =  $(NVCD_HOME))





