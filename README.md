## libnvcd
Nvidia CUDA Dump

============
This is an easy-to-use, performance measurement tool for NVIDIA based GPUs. The tool queries CUPTI APIs for reading both events and metrics for functions or selected regions in a GPU (or CPU+GPU) code. As of now, the tools only report the events and metrics that CUPTI provides. In future, we will add a separate analysis module that combines these events and metrics to compute our own derived measures.

There are mainly two tools:
1. Source-code based annotation library
2. Standalone tool for automatically finding groups of counters available on a given firmware. 


## Compile

make clean && make libnvcd.so && make nvcdrun

## Use in a source code


## Using the standalone tool
