#!/bin/bash

rm -f *.log

mpirun -n $1 ./producer_consumer $2
