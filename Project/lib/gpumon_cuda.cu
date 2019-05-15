#include "gpumon.h"

#include "util.h"
#include <stdlib.h>
#include <time.h>
#include <vector>

static __device__ clock64_t* dev_tstart = nullptr;
static __device__ clock64_t* dev_ttime = nullptr;
static size_t dev_tbuf_size = 0;

static __device__ int* dev_num_iter = nullptr;
static size_t dev_num_iter_size = 0;

//static __device__ int* dev_number = nullptr;

EXTC HOST void gpumon_init_device_mem(int num_threads) {
	{
		void* dev_tstart_addr = nullptr;
		CUDA_RUNTIME_FN(cudaGetSymbolAddress(&dev_tstart_addr, dev_tstart));

		void* dev_ttime_addr = nullptr;
		CUDA_RUNTIME_FN(cudaGetSymbolAddress(&dev_ttime_addr, dev_ttime));
	
		dev_tbuf_size = sizeof(clock64_t) * static_cast<size_t>(num_threads);
	
		CUDA_RUNTIME_FN(cudaMalloc(&dev_tstart_addr, dev_tbuf_size));
		CUDA_RUNTIME_FN(cudaMalloc(&dev_ttime_addr, dev_tbuf_size));
	}

	// test code
	{
		void* dev_num_iter_addr = nullptr;
		CUDA_RUNTIME_FN(cudaGetSymbolAddress(&dev_num_iter_addr, dev_num_iter));

		//	void* dev_num_size = nullptr;
		
		dev_num_iter_size = sizeof(int) * static_cast<size_t>(num_threads);

		CUDA_RUNTIME_FN(cudaMalloc(&dev_num_iter, dev_num_iter_size));
		
		
		std::vector<int> host_num_iter(num_threads, 0);

		int iter_min = 1000000;
		int iter_max = iter_min * 100;
			
		for (size_t i = 0; i < host_num_iter.size(); ++i) {
			srand(time(nullptr));
			host_num_iter[i] = iter_min + (rand() % (iter_max - iter_min));
		}

		CUDA_RUNTIME_FN(cudaMemcpy(dev_num_iter_addr,
															 &host_num_iter[0],
															 dev_num_iter_size,
															 cudaMemcpyHostToDevice));
	}
}

EXTC HOST void gpumon_get_device_ttime(clock64_t* out) {
	CUDA_RUNTIME_FN(cudaMemcpyFromSymbol(out, "dev_ttime", dev_tbuf_size));
}


EXTC DEV void gpumon_device_start(int thread) {
	dev_tstart[thread] = clock64(); 
}

EXTC DEV void gpumon_device_end(int thread) {
	dev_ttime[thread] = clock64() - dev_tstart[thread]; 
}

EXTC GLOBAL void gpumon_kernel_test(int num_threads) {
	int thread = threadIdx.x;

	if (thread < num_threads) {
		gpumon_device_start(thread);
		
		volatile int number = 0;
		for (int i = 0; i < dev_num_iter[thread]; ++i) {
			number += i;
		}

		gpumon_device_end(thread);
	}
}

EXTC HOST void gpumon_kernel_test_call(int num_threads) {
	dim3 block(num_threads, 1, 1);
	dim3 grid(1, 1, 1);

	gpumon_kernel_test<<<grid, block>>>(1024);
}
