#!/bin/bash

rm -f *.log

mpirun -n $1 ./workpool $2 > wp_output_${1}_${2}.log

mkdir -p "wp_log${1}"
mv *.log "wp_log${1}"
