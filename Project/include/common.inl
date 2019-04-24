/*
 * provides a random number in the range [min, max)
 */

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
			ss << std::to_string(in[i]);

			if (i < n - 1) {
				ss << ", ";
			}
		}

		ss << " }";

		return ss.str();
	}

}
