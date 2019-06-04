#!/bin/bash

#SBATCH --job-name="run-nvcdrun"
#SBATCH --output="/home/schutth/cs415/Project/xsede-scripts/job-output/run-nvcdrun.%j.%N.out"
#SBATCH --workdir="/home/schutth/cs415/Project"
#SBATCH --partition=gpu-shared
#SBATCH --gres=gpu:k80:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH -t 00:10:00

echo "env preserved? XSEDE_ENV_PATH = ${XSEDE_ENV_PATH}" 

source xsede-scripts/xsede-env.sh

make clean && make libnvcd.so && make nvcdrun

retVal=$?

if [ $retVal -ne 0 ]; then
    export BENCH_EVENTS="ALL"
    bin/nvcdrun
else
    printf "\nCompilation failure...\n"
    exit $retVal
fi
