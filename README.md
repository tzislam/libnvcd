## libnvcd
This is an easy-to-use, performance measurement tool for NVIDIA based GPUs. The tool queries CUPTI APIs for reading both events and metrics for functions or selected regions in a GPU (or CPU+GPU) code. As of now, the tools only report the events and metrics that CUPTI provides. In future, we will add a separate analysis module that combines these events and metrics to compute our own derived measures.

*Currently, libNVCD only supports Nvidia GPUs and CUDA.*
** This tool is actively being developed. We will try out best to keep the current API as is. However, there is no guarantee. Please use with caution. **

There are mainly two tools:
* 1. Source-code based annotation library
* 2. Standalone tool for automatically finding groups of counters available on a given firmware. 


## Compile

make clean && make libnvcd.so && make nvcdrun

## How to use in a source code

There are two different granularities at which libNVCD can be used--(a) function level, and (b) wherever you want.


## Using the standalone query tool for listing all possible groups of events and metrics to list


## Output format


## What is not recorded by this tool




## How to cite this work
Please use the following citation:
```
@Misc{holland20libnvcd,
  title =        {libNVCD: A per-thread hardware performance counter measurement tool for GPUs},
  author = {Schutte, Holland and Islam, Tanzima Z.},
  year = {2020},
  note =         {\url{https://github.com/tzislam/libnvcd}}
}
```
