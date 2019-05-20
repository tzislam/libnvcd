#!/bin/bash

rm -f *.log

mpirun -n $1 ./producer_consumer $2 > pc_output_${1}_${2}.log

mkdir -p "log${1}"
mv *.log "log${1}"
