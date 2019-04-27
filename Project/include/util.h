#ifndef __UTIL_H__
#define __UTIL_H__

#include "commondef.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <stdlib.h>
#include <time.h>

#include <string>
#include <sstream>

extern "C" {

	void cuda_runtime_error_print_exit(cudaError_t status,
									   int line,
									   const char* file,
									   const char* expr);

	void cuda_driver_error_print_exit(CUresult status,
									  int line,
									  const char* file,
									  const char* expr);
	
	void cupti_error_print_exit(CUptiResult status,
								int line,
								const char* file,
								const char* expr);
}

namespace util {

	/*
	 * provides a random number in the range [min, max)
	 */
	
	template <typename scalarType>
	scalarType random_next(scalarType min, scalarType max)
	{
		srand(time(NULL));

		return max + rand() % (max - min);
	}

	template <typename type>
	void cuda_malloc_or_die(type** out, size_t sz)
	{
		CUDA_RUNTIME_FN(cudaMalloc(static_cast<void**>(&out), sz));
	}
	
	template <typename type>
	std::string to_string(const type* in, size_t n)
	{
		std::stringstream ss;

		ss << "{ ";
	
		for (size_t i = 0; i < n; ++i) {
			type x = in[i];
			ss << std::to_string(x);

			if (i < n - 1) {
				ss << ", ";
			}
		}

		ss << " }";

		return ss.str();
	}

}

#endif // __UTIL_H__
