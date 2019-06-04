#!/bin/bash

srun --partition=gpu-shared --gres=gpu:k80:1 --pty --nodes=1 --ntasks-per-node=4 -t $1:00:00 --wait=0 --export=ALL /bin/bash
