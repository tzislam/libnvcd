#!/bin/bash

nproc=$1
layout=$2

if [ -z "$nproc" ]; then
    nproc=4;
fi

echo "number of procs is set to ${nproc}"

if [ -z "$layout" ]; then
    layout=2
fi

echo "file layout id is set to ${layout} (see fileinfo.h)"

rm -f outputfile
mpicc -DLAYOUT=$layout -g -ggdb write_all_lab.c -o write
mpicc -DLAYOUT=$layout -g -ggdb read_all_lab.c -o read

printf "---write---"
mpirun -n $nproc ./write
cat outputfile
printf "\n"

printf "---read---"
mpirun -n $nproc ./read > readout
printf "readout..."
cat readout
printf "\n"
