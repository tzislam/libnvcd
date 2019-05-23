#!/bin/bash

srun --partition=debug --pty --nodes=1 --ntasks-per-node=4 -t 00:10:00 --wait=0 --export=ALL /bin/bash
