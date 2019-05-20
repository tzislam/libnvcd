#!/bin/bash

rm -f *.log
rm -rf sim*

t=120

for i in 1 2 3 4 5
do
    mkdir -p "wp_sim${i}"
    
    cd "wp_sim${i}"

    echo "on wp_sim ${i}"
    
    for k in 4 8 12 16
    do
        echo "\trunning with n=${k}"
        
        mpirun -n $k ../workpool $t > ./wp_output_${k}_${t}.log

        mkdir -p "wp_log${k}"
        mv *.log "wp_log${k}"
    done

    cd ../
done
