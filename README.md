## libnvcd
This is an easy-to-use, performance measurement tool for NVIDIA based GPUs. The tool queries CUPTI APIs for reading both events and metrics for functions or selected regions in a GPU (or CPU+GPU) code. As of now, the tools only report the events and metrics that CUPTI provides. In future, we will add a separate analysis module that combines these events and metrics to compute our own derived measures.

*Currently, libNVCD only supports Nvidia GPUs and CUDA.*

**This tool is actively being developed. We will try out best to keep the current API as is. However, there is no guarantee. Please use with caution.**

There are mainly two tools:
* Source-code based annotation library
* Standalone tool for automatically finding groups of counters available on a given firmware. 


## Compile

make clean && make libnvcd.so && make nvcdrun

## How to use in a source code

There are two different granularities at which libNVCD can be used--(a) function level, and (b) wherever you want.


## Using the standalone query tool for listing all possible groups of events and metrics to list

The standalone tool *nvcdinfo* can be used to automatically put the metrics and events into groups of user-specified sizes. 
The output from this tool allows a user to generate a list of groups of metrics or events that can be collected in each pass. Since not all metrics and events can be collected at once due to hardware limitation, this tool can be used first to estimate the number of passes one would need to collect all (or a selected subset of) metrics. Also, not all metrics or events can be collected together due to hardware resource conflict. This tool also helps address this issue -- it automatically generated groups of metrics and events that CAN be collected together.

Usage 1: 
./nvcdinfo [-n GROUP_SIZE] -d DEVICE_ID
Generates a list of metrics for DEVICE_ID (i.e., gpu id. For a node with 4 gpus, DEVICE_ID will be between 0 and 3 inclusive). Each entry in the output will pertain to a group of metrics that can be counted and each group will have at most GROUP_SIZE number of metrics. This does not guarantee that each group will indeed have all of the GROUP_SIZE number of metrics. The actual number of metrics per group depends on the availability of that many metrics per group. By default, the GROUP_SIZE is set to 1, meaning one metric per group. We recommend using a large value for GROUP_SIZE, such as 100 to ensure the largest possible groups to reduce the number of passes needed to cover collecting all metrics.

Usage 2: 
./nvcdinfo [-n GROUP_SIZE] -d DEVICE_ID -e
Generates a list of events for DEVICE_ID (i.e., gpu id. For a node with 4 gpus, DEVICE_ID will be between 0 and 3 inclusive). Each entry in the output will pertain to a group of events that can be counted and each group will have at most GROUP_SIZE number of events. This does not guarantee that each group will indeed have all of the GROUP_SIZE number of events. The actual number of events per group depends on the availability of that many events per group. The flag "-e" means the user wants to list all events. By default, the GROUP_SIZE is set to 1, meaning one event per group. We recommend using a large value for GROUP_SIZE, such as 100 to ensure the largest possible groups to reduce the number of passes needed to cover collecting all events.

Usage 3: 
./nvcdinfo -d DEVICE_ID -m
Generates a list of all metrics and the events used to calculate those metrics for DEVICE_ID (i.e., gpu id. For a node with 4 gpus, DEVICE_ID will be between 0 and 3 inclusive). This map of metrics to events can be useful for postprocessing later.


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
