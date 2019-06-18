#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVCD_HOME/bin
export BENCH_EVENTS="branch:divergent_branch:threads_launched:warps_launched"
export BENCH_METRICS="branch_efficiency:stall_constant_memory_dependency:stall_exec_dependency:stall_inst_fetch:stall_memory_dependency:stall_memory_throttle:warp_execution_efficiency:warp_nonpred_execution_efficiency"
