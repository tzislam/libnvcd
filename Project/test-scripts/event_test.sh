#!/bin/bash

# all tests ran against CUDA 9.2

case $1 in
    # k80, 3.7: should fail
    1) export BENCH_EVENTS="tex_cache_hit:branch:divergent_branch"
        ;;

    # k80, 3.7: should succeed
    2) export BENCH_EVENTS="sm_cta_launched" 
        ;;

    # k80, 3.7: should fail, despite being defined for 2.x AND 1.x 
    3) export BENCH_EVENTS="branch" 
        ;;

    # k80, 3.7: should succeed, but counters are all 0
    4) export BENCH_EVENTS="l1_local_load_hit:l1_global_load_hit:l1_local_load_miss:l1_global_load_miss"
       ;;

    # k80, 3.7: should succeed, with all events reported.
    5) export BENCH_EVENTS="l1_local_load_hit:l1_global_load_hit:l1_local_load_miss:l1_global_load_miss:ALL"
       ;;

    6) export BENCH_EVENTS="ALL"
       ;;

    # used in experiments for presentation
    7) export BENCH_EVENTS="l1_global_load_miss:l2_subp0_write_sector_misses:l2_subp0_read_sector_misses:active_warps:branch:divergent_branch"
       export BENCH_METRICS="branch_efficiency:sm_efficiency:ipc:l1_cache_global_hit_rate:dram_read_throughput:dram_write_throughput"
       ;;
    
    *)
        echo "Invalid argument specified"
        exit 1
        ;;
esac

dt=$(date '+%m-%d-%Y_%H-%M-%S')

d=tests/event_${1}

rm -f $d/*

mkdir -p $d

./run > "${d}/out_${1}_${dt}"
