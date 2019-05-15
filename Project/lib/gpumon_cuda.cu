#include "gpumon.h"

#include "util.h"
#include <stdlib.h>
#include <time.h>
#include <vector>

__device__ clock64_t* dev_tstart;
__device__ clock64_t* dev_ttime;
__device__ int* dev_num_iter;

static size_t dev_tbuf_size = 0;
static size_t dev_num_iter_size = 0;

static void* d_dev_tstart = nullptr;
static void* d_dev_ttime = nullptr;
static void* d_dev_num_iter = nullptr;

template <class T>
void* __cuda_zalloc_sym(size_t size, const T& sym, const char* ssym) {
	void* address_of_sym = nullptr;
	CUDA_RUNTIME_FN(cudaGetSymbolAddress(&address_of_sym, sym));

	void* device_allocated_mem = nullptr;
	
	CUDA_RUNTIME_FN(cudaMalloc(&device_allocated_mem, size));
	CUDA_RUNTIME_FN(cudaMemset(device_allocated_mem, 0, size));
	
	CUDA_RUNTIME_FN(cudaMemcpy(address_of_sym,
														 &device_allocated_mem,
														 sizeof(device_allocated_mem),
														 cudaMemcpyHostToDevice));

	return device_allocated_mem;
}
#define cuda_zalloc_sym(sz, sym) __cuda_zalloc_sym(sz, sym, #sym)

template <class T>
void cuda_safe_free(T*& ptr) {
	if (ptr != nullptr) {
		CUDA_RUNTIME_FN(cudaFree(static_cast<void*>(ptr)));
	}
}


EXTC HOST void gpumon_free_device_mem() {
	cuda_safe_free(d_dev_tstart);
	cuda_safe_free(d_dev_ttime);
	cuda_safe_free(d_dev_num_iter);
}

EXTC HOST void gpumon_init_device_mem(int num_threads) {
	{	
		dev_tbuf_size = sizeof(clock64_t) * static_cast<size_t>(num_threads);

		d_dev_tstart = cuda_zalloc_sym(dev_tbuf_size, dev_tstart);

		d_dev_ttime = cuda_zalloc_sym(dev_tbuf_size, dev_ttime);
	}

	// test code
	{
		dev_num_iter_size = sizeof(int) * static_cast<size_t>(num_threads);

		d_dev_num_iter = cuda_zalloc_sym(dev_num_iter_size, dev_num_iter);
		
		std::vector<int> host_num_iter(num_threads, 0);

		int iter_min = 1000000;
		int iter_max = iter_min * 100;
			
		for (size_t i = 0; i < host_num_iter.size(); ++i) {
			srand(time(nullptr));
			host_num_iter[i] = iter_min + (rand() % (iter_max - iter_min));
		}

		CUDA_RUNTIME_FN(cudaMemcpy(d_dev_num_iter,
															 &host_num_iter[0],
															 dev_num_iter_size,
															 cudaMemcpyHostToDevice));
	}
}

EXTC HOST void gpumon_get_device_ttime(clock64_t* out) {	
	CUDA_RUNTIME_FN(cudaMemcpy(out,
														 d_dev_ttime,
														 dev_tbuf_size,																			 
														 cudaMemcpyDeviceToHost));
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
