## libnvcd
This is an easy-to-use, performance measurement tool for NVIDIA based GPUs. The tool queries CUPTI APIs for reading both events and metrics for functions or selected regions in a GPU (or CPU+GPU) code. As of now, the tool only reports the events and metrics that CUPTI provides. In future, we will add a separate analysis module that combines these events and metrics to compute our own derived measures.

*Currently, libNVCD only supports Nvidia GPUs and CUDA.*

<!--**This tool is actively being developed. We will try our best to keep the current API as is. However, there is no guarantee. Please use with caution.**-->

There are mainly two tools:
* Source-code based annotation and interception library
* Standalone tool for automatically finding groups of counters available on a given firmware. 


## Compile

`make clean && make [DEBUG=1] [libnvcdhook.so] [nvcdrun] [nvcdinfo]`

Each of the optional targets specified in the command above builds `libnvcd.so` first.

These executables will build in the `bin` directory that lies at the root of the repo.

If `DEBUG=1` is provided, optimizations are turned off and debug symbols are provided.

Note also that environment variables `CUDA_HOME` and `CUDA_ARCH_SM` need to be set - see their defaults in `Makefile.inc` for an example.

`CUDA_HOME` should point to the very root directory of the cuda installation, and `CUDA_ARCH_SM` should be provided in the form of `sm_<version>`.

The user can either edit this manually in `Makefile.inc`, or define them as environment variables (in which case they will override the defaults).

## How it works

- We hook the cuda API function, `cudaLaunchKernel()`.

- Each time a kernel is invoked, `cudaLaunchKernel()` is called.

= When `cudaLaunchKernel()` is called, libnvcd will only record event counters if the kernel invocation lies within an annotated region.

- `libnvcd_begin()` and `libnvcd_end()` mark the start and end of a region, respectively. 

- These functions are loaded at runtime through the hook's library, which is designed to be loaded using `LD_PRELOAD`. Thus, there is no need to link against `libnvcd.so` _unless_ the user
doesn't want to leverage the hook functionality. <!-- (though this will require more work). --> 

- Region annotation may contain invocations for multiple kernels or a single kernel - it's up to the user. As is the name of the region itself.

- Whatever counters have been specified by the user will be recorded by a callback within `libnvcd.so` that interacts with the CUPTI Event and Callback APIs.

## How to use in a source code

### nvcdinfo

You can first use `nvcdinfo` to determine what counters are available on the system. 

The script `nvcdinfo_gpu0` provides an example that fetches counters for the first GPU in the system. 

From there, it will spit out csv files for each event domain that the GPU has available. The simplest approach is to take one of the lines in the CSV file and use that line as the event counters you wish to record.

### BENCH_EVENTS

By using the desired line of event counters and setting them as the value to the `BENCH_EVENTS` environment variable, each counter will be recorded. 

For example, on some systems the following line will be present in one of the CSV files:

`branch,divergent_branch,threads_launched,warps_launched`

Simply taking this line and setting it as follows is enough:

`export BENCH_EVENTS=branch,divergent_branch,threads_launched,warps_launched`

The reason why we group these accordingly is because only a very specific set of events may be used together at once. Only one of these event groups can be recorded per kernel invocation, so

it's simplest to stick with recording on a per-line basis. That said, the library does support the usage of incompatible events and will find separate groups to section them off with.

### Libraries

You must ensure that `LD_PRELOAD` contains the path to `libnvcdhook.so`, and that `LD_LIBRARY_PATH` points to the `bin` directory in the repo. It may also need to be set to point to cuda and cupti's locations.

### Source code

From there, make sure to add `-I/path/to/libnvcd-repo/include` to your cpreprocessor flags, `#include <libnvcd.h>` on any source files with cuda calls you wish to use, 
and then call `libnvcd_init()`. This will load `libnvcd_begin()` and `libnvcd_end()`. From there, mark your regions and you'll get output.

See `nvcdrun/src/gpu_call.cu` for an in source example.

## Current limiations

### Not yet implemented

- The end goal of this project is to be compatible with MPI and multi-threaded (with one thread per GPU). Currently, neither of these are supported.

- `nvcdinfo` only provides information on event counters, not metrics.

### IBM LSF, JSM

The systems that this library has been tested on primarily are HPC clusters that are built off of IBM's LSF, and use the `jsrun` resource allocation command to perform work.

We've experienced some issues with `jsrun` on one machine so far, but on the other the issue is non-existent. Launching an interactive session and ensuring that you have private

access to all available GPUs on the node should be sufficient to get work done. 

### CUDA_VISIBLE_DEVICES

If it is the case that you do _not_ have access to all GPUs, you can set the environment

variable `CUDA_VISIBLE_DEVICES` to the ID of the GPU you wish to use. This will _map_ the physical ID of the GPU to a "virtual" ID of 0 that is known and used by the program.

It is recommended that, for now, you use 1 GPU per invocation and use `CUDA_VISIBLE_DEVICES` to control which GPU is in use. 

For example, if the machine you're working on has 4 GPUs available, each numbered 0-3, and you do:

`export CUDA_VISIBLE_DEVICES=1`, and then run `nvcdrun`, it will only see the device ID'd through index 1 on the GPU array. As far as the CUDA driver is concerned,

querying for GPU 0 will actually return GPU 1.

## Using the standalone query tool for listing all possible groups of events and metrics to list

The standalone tool *nvcdinfo* can be used to automatically put the metrics and events into groups of user-specified sizes. 
The output from this tool allows a user to generate a list of groups of events that can be collected in each pass. Since not all metrics and events can be collected at once due to hardware limitation, this tool can be used first to estimate the number of passes one would need to collect all (or a selected subset of) metrics. Also, not all metrics or events can be collected together due to hardware resource conflict. This tool also helps address this issue -- it automatically generated groups of events that CAN be collected together.

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

We currently support collecting metrics and events. Metrics are specified in the exact same way events are, but through the `BENCH_METRICS` environment variable.
Soon we will provide better auxilary support by enabling *nvcdinfo* to report on groups of metrics. That said, if you know what metrics are available on your system, you will get the information that you seek
by setting `BENCH_METRICS` accordingly.


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
