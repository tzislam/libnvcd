#!/bin/bash
#SBATCH --job-name="matmulpdump"
#SBATCH --output="matmulpdump.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
#SBATCH -t 00:05:00
export OMP_NUM_THREADS=24
export PDUMP_EVENTS=PAPI_L1_DCM
module load hdf5
module load papi
#export SLURM_NODEFILE=`generate_pbs_nodefile`
ibrun --npernode 1 ./perf-dump.bin