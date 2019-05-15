#include "gpumon.h"

#include "util.h"
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <unordered_map>

//--------------------------------------
// internal
//-------------------------------------

DEV clock64_t* dev_tstart;
DEV clock64_t* dev_ttime;
DEV int* dev_num_iter;
DEV uint* dev_smids;

static size_t dev_tbuf_size = 0;
static size_t dev_num_iter_size = 0;
static size_t dev_smids_size = 0;

static void* d_dev_tstart = nullptr;
static void* d_dev_ttime = nullptr;
static void* d_dev_num_iter = nullptr;
static void* d_dev_smids = nullptr;

template <class T>
static void* __cuda_zalloc_sym(size_t size, const T& sym, const char* ssym) {
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
static void cuda_safe_free(T*& ptr) {
	if (ptr != nullptr) {
		CUDA_RUNTIME_FN(cudaFree(static_cast<void*>(ptr)));
	}
}

template <class T>
static void cuda_memcpy_host_to_dev(void* dst, std::vector<T> host) {
	size_t size = host.size() * sizeof(T);
	
	CUDA_RUNTIME_FN(cudaMemcpy(static_cast<void*>(dst),
														 static_cast<void*>(host.data()),
														 size,
														 cudaMemcpyHostToDevice));

}

// see https://devtalk.nvidia.com/default/topic/481465/any-way-to-know-on-which-sm-a-thread-is-running-/
DEV uint get_smid() {
	uint ret;
	asm("mov.u32 %0, %smid;" : "=r"(ret) );
	return ret;
}

struct kernel_invoke_data {
	struct thread_slice_info {
		unsigned smid;
		std::vector<int> thread_indices;

		thread_slice_info(unsigned smid_, std::vector<int> thread_indices_)
			: smid(smid_),
				thread_indices(std::move(thread_indices_))
		{}
	};
	
	using time_thread_map_t = std::unordered_map<clock64_t, thread_slice_info>;

	time_thread_map_t time_to_thread_info;

	
};

//-------------------------------------
// public
//-------------------------------------

EXTC HOST void gpumon_free_device_mem() {
	cuda_safe_free(d_dev_tstart);
	cuda_safe_free(d_dev_ttime);
	cuda_safe_free(d_dev_num_iter);
	cuda_safe_free(d_dev_smids);
}

EXTC HOST void gpumon_init_device_mem(int num_threads) {
	{	
		dev_tbuf_size = sizeof(clock64_t) * static_cast<size_t>(num_threads);

		d_dev_tstart = cuda_zalloc_sym(dev_tbuf_size, dev_tstart);
		d_dev_ttime = cuda_zalloc_sym(dev_tbuf_size, dev_ttime);
	}

	{
		dev_smids_size = sizeof(uint) * static_cast<size_t>(num_threads);
		
		d_dev_smids = cuda_zalloc_sym(dev_smids_size, dev_smids);
	}

	// test code
	{
		dev_num_iter_size = sizeof(int) * static_cast<size_t>(num_threads);

		d_dev_num_iter = cuda_zalloc_sym(dev_num_iter_size, dev_num_iter);
		
		std::vector<int> host_num_iter(num_threads, 0);

		int iter_min = 1000;
		int iter_max = iter_min * 100;
			
		for (size_t i = 0; i < host_num_iter.size(); ++i) {
			srand(time(nullptr));
			host_num_iter[i] = iter_min + (rand() % (iter_max - iter_min));
		}

		cuda_memcpy_host_to_dev<int>(d_dev_num_iter, std::move(host_num_iter));
	}
}

EXTC HOST void gpumon_get_device_ttime(clock64_t* out) {	
	CUDA_RUNTIME_FN(cudaMemcpy(out,
														 d_dev_ttime,
														 dev_tbuf_size,																			 
														 cudaMemcpyDeviceToHost));
}

EXTC HOST void gpumon_get_device_smids(unsigned* out) {
	CUDA_RUNTIME_FN(cudaMemcpy(out,
														 d_dev_smids,
														 dev_smids_size,																			 
														 cudaMemcpyDeviceToHost));

}

EXTC DEV void gpumon_device_start(int thread) {
	dev_tstart[thread] = clock64(); 
}

EXTC DEV void gpumon_device_end(int thread) {
	dev_ttime[thread] = clock64() - dev_tstart[thread];
	dev_smids[thread] = get_smid();
}

EXTC GLOBAL void gpumon_kernel_test() {
	int thread = threadIdx.x;
	
	if (thread < blockDim.x) {
		gpumon_device_start(thread);
		
		volatile int number = 0;

		for (int i = 0; i < dev_num_iter[thread]; ++i) {
			number += i;
		}

		gpumon_device_end(thread);
	}
}

EXTC HOST void gpumon_kernel_test_call(int num_threads) {
	gpumon_kernel_test<<<1, num_threads>>>();
}
