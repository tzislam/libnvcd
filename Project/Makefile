#!/bin/bash

CC=nvcc
CUDACC=nvcc

$CUDACC -c -I./include -arch=sm_50 src/gpu.cu -o obj/gpu.o &&\
    $CC -v -I./include -L/usr/lib/x86_64-linux-gnu -lnvidia-ml -lcuda -lcudart obj/gpu.o src/main.cpp -o bin/perfmon

#gcc-7 -v -lnvidia-ml -lcuda -lcudart src/main.cpp -o bin/perfmon

