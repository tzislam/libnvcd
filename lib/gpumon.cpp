#include "gpumon.h"
#include "gpumon.cuh"
#include "util.h"

#include <vector>
#include <algorithm>
#include <iterator>
#include <stdio.h>

struct block {
	int thread;
	clock64_t time;
};

struct kernel_invoke_data {
	std::vector<block> load_minor;
	std::vector<block> load_major;
	
	std::vector<clock64_t> times;
	std::vector<unsigned> smids;
	
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

	void fill_outliers(double bounds,
										 double q1,
										 double q3,
										 const std::vector<block>& in,
										 std::vector<block>& out) {
		clock64_t max = static_cast<clock64_t>(q3)
			+ static_cast<clock64_t>(bounds);

		printf("{q1, q3, bounds, max} = {%f, %f, %f, %llu}\n",
					 q1, q3, bounds, max);

		for (const block& b: in) {
			if (b.time > max) {
				out.push_back(b);
			}
		}
	}

	void print_blockv(const char* name, const std::vector<block>& v) {
		printf("=====%s=====\n", name);
		
		for (size_t i = 0; i < v.size(); ++i) {
			printf("[%lu] time = %llu, thread = %i\n",
						 i,
						 v[i].time,
						 v[i].thread);
		}
	}
	
	void write() {
		for (size_t i = 0; i < num_threads; ++i) {
			printf("[%lu] time = %lli, smid = %i\n",
						 i,
						 times[i],
						 smids[i]);
		}
		
		{
			std::vector<block> sorted;

			for (size_t i = 0; i < num_threads; ++i) {
				sorted.push_back(block{static_cast<int>(i), times[i]}); 
			}
			
			std::sort(sorted.begin(), sorted.end(), [](const block& a, const block& b) -> bool {
					return a.time < b.time;
				});

			print_blockv("sorted", sorted);

			size_t qlen = num_threads >> 2;

			double q1 = static_cast<double>(sorted[qlen].time)
				+ static_cast<double>(sorted[qlen - 1].time);
			q1 = q1 * 0.5;

			double q3 = static_cast<double>(sorted[qlen * 3].time)
				+ static_cast<double>(sorted[(qlen * 3) - 1].time);
			q3 = q3 * 0.5;

			double iqr = q3 - q1;

			double minorb = iqr * 1.5;
			fill_outliers(minorb, q1, q3, sorted, load_minor);

			double majorb = iqr * 3.0;
			fill_outliers(majorb, q1, q3, sorted, load_major);
		}

		print_blockv("load_minor", load_minor);
		print_blockv("load_major", load_major);
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

