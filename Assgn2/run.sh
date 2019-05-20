#!/bin/bash

rm -f *.log

t=120

for i in 1 2 3 4 5
do
    mkdir -p "pc_sim${i}"
    
    cd "pc_sim${i}"

    echo "on sim ${i}"
    
    for k in 4 8 12 16
    do
        echo "\trunning with n=${k}"
        
        mpirun -n $k ../producer_consumer $t > ./pc_output_${k}_${t}.log

        mkdir -p "pc_log${k}"
        mv *.log "pc_log${k}"
    done

    cd ../
done
