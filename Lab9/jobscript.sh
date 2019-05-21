#!/bin/bash
#SBATCH --job-name="hellompirunrsh"
#SBATCH --output="hellompirunrsh.%j.%N.out"
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
#SBATCH -t 11:35:00
export MV2_SHOW_CPU_BINDING=1 #shows the job binding
ibrun -np 8 ./kripke dfg sdfg sdfg
