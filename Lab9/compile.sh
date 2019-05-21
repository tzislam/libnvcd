#!/bin/bash

mpic++ -I./perf-dump -openmp -o test matmul.c ./perf-dump/libperfdump.a /opt/hdf5/intel/mvapich2_ib/lib/libhdf5.a /opt/papi/intel/lib/libpapi.a -lz -lmpi
