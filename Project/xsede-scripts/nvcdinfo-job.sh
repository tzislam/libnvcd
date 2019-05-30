#!/bin/bash
#SBATCH --job-name="nvcdinfo"
#SBATCH --output="nvcdinfo.log"
#SBATCH --partition=gpu-shared
#SBATCH --gres=gpu:k80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH -t 00:05:00

source xsede-scripts/xsede_env.sh
bin/nvcdinfo
