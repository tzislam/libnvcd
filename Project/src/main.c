#include "commondef.h"
#include "gpu.h"
#include <assert.h>

#define ENV_METRICS "ENV_METRICS"
#define ENV_DELIM ':'

struct options {
	char** metrics;
	size_t num_metrics;
};


struct options g_options = {0};

size_t count_env_list(const char* str)
{
	char* list = getenv(ENV_METRICS);
	size_t count = 0;
	
	if (list != NULL) {
		size_t length = strlen(list);

		char* p = list;

		while (*p && *p != '=') {
			p++;
		}

		assert(*p == '=');

		int scanning = 1;
		char* delim = strchr(p, ENV_DELIM);
		
		while (scanning) {

			while (*p && *p != ENV_DELIM) {
				p++;
			}

			count++;

			scanning = delim == NULL || *p == '\0';
			
			if (delim != NULL) {
				delim = strchr(delim + 1, ENV_DELIM);
			}
		}
	} 
}

void getenv_list()
{
}

int main()
{
	gpu_test_matrix_vec_mul();
	
	gpu_test();
	
	CUDA_RUNTIME_FN(cudaDeviceSynchronize());

	return 0;
}
