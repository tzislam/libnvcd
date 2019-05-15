#include "gpumon.h"
#include "gpumon.cuh"
#include "util.h"

#include <vector>
#include <algorithm>
#include <iterator>
#include <stdio.h>

struct kernel_invoke_data {
	std::vector<clock64_t> times;
	std::vector<unsigned> smids;

	std::vector<int> outlier_threads;
	
	double time_stddev;
	double time_mean;

	size_t num_threads;
	
	kernel_invoke_data(size_t num_threads_)
		: times(num_threads_, 0),
	    smids(num_threads_, 0),
			time_stddev(0.0),
			num_threads(num_threads_)
	{}

	~kernel_invoke_data()
	{}
	
	void write() {
		for (size_t i = 0; i < num_threads; ++i) {
			printf("[%lu] time = %lli, smid = %i\n",
						 i,
						 times[i],
						 smids[i]);
		}

		#if 0
		{
			double mean = 0.0;

			size_t qlen = num_threads >> 2;

			std::vector<clock64_t> v;
			std::copy(times.begin(), times.end(), std::back_inserter(v));

			std::sort(v.begin(), v.end());

			
		}
		#endif
	}
};

static std::vector<kernel_invoke_data> kernel_invoke_list;
static int num_threads = 0;

EXTC HOST void gpumon_host_start(int n_threads) {
	CUDA_RUNTIME_FN(cudaSetDevice(0));

	num_threads = n_threads;
	
	gpumon_init_device_mem(num_threads);
}

EXTC HOST void gpumon_host_end() {
	CUDA_RUNTIME_FN(cudaDeviceSynchronize());

	kernel_invoke_data d(static_cast<size_t>(num_threads));
	
	gpumon_get_device_ttime(&d.times[0]);
  gpumon_get_device_smids(&d.smids[0]);

	d.write();

	kernel_invoke_list.push_back(std::move(d));
	
	gpumon_free_device_mem();
}

