#ifndef __COMMON_INL__
#define __COMMON_INL__

/*
 * provides a random number in the range [min, max)
 */

#ifndef __CUDA_ARCH__

#include <cuda.h>

#include <stdlib.h>
#include <time.h>

#include <string>
#include <sstream>

#pragma message "YEET"

namespace util {

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
#endif

#endif // __COMMON_INL__
