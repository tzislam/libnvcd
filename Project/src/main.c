#include "commondef.h"
#include "gpu.h"

#include <assert.h>
#include <ctype.h>
//#include <stdlib.h>

#define ENV_METRICS "ENV_CUPTI_METRICS"
#define ENV_DELIM ':'

void* zalloc(size_t sz)
{
	void* p = malloc(sz);

	if (p != NULL) {
		memset(p, 0, sz);
	} else {
		puts("OOM");
	}

	/* set here for testing purposes; 
	   should not be relied upon for any
	   real production build */
	assert(p != NULL);

	return p;
}

const char* env_var_list_start(const char* list)
{
	const char* p = list;

	while (*p && *p != '=') {
		p++;
	}

	assert(*p == '=');

	const char* ret = p + 1;

	if (!isalpha(*ret)) {
		printf("ERROR: %s must begin with a letter.\n", ret);
		ret = NULL;
	}

	return ret;
}

const char* env_var_list_scan_entry(const char* p, size_t* p_count)
{
	size_t count = 0;

	bool error = false;
	
	while (*p && *p != ENV_DELIM && !error) {
		error = !isalnum(*p) && !(*p == '_');
		
		if (error) {
			printf("ERROR: invalid character found: %s.\n", p);
		} else {
			count++;
			p++;
		}
	}

	if (p_count != NULL) {
		*p_count = count;
	}

	if (error) {
		p = NULL;
	}

	return p;
}

typedef int (*env_var_list_scan_fn_t)(const char* entry, size_t entry_len, void* user);

typedef void (*env_var_list_scan_error_fn_t)(void* user);

struct env_var_list_scan_ctx {
	char** list;
	size_t index;
	size_t num_elems;
};

int env_var_list_count_entry(const char* entry, size_t entry_len, void* user)
{
	assert(user != NULL);
	size_t* count = (size_t*) user;
	*count = *count + 1;
	return 1;
}

void env_var_list_count_entry_error(void* user)
{
	assert(user != NULL);

	size_t* count = (size_t*) user;
	 *count = 0;
}

int env_var_list_insert_entry(const char* entry, size_t entry_len, void* user)
{
	assert(user != NULL);
	struct env_var_list_scan_ctx* ctx = (struct env_var_list_scan_ctx*) user;
	
	char* str = zalloc((entry_len + 1) * sizeof(char));

	assert(ctx->index < ctx->num_elems);
	
	if (str != NULL) {
		strncpy(str, entry, entry_len);

		ctx->list[ctx->index] = str;
		ctx->index++;
	}

	return str != NULL;
}

void env_var_list_scan(const char* var,
											 env_var_list_scan_fn_t callback,
											 env_var_list_scan_error_fn_t error,
											 void* user)
{
	const char* p = env_var_list_start(var);

	if (p != NULL) {
		const char* delim = strchr(p, ENV_DELIM);

		bool scanning = *p != '\0';
		
		while (scanning) {
			size_t this_count = 0;

			if (env_var_list_scan_entry(p, &this_count) == NULL) {
				scanning = false;
			} else {
				scanning = this_count != 0;
			}
		
			if (scanning) {
				scanning = callback(p, this_count, user);
				if (scanning) {
					if (delim != NULL) {
						p = delim + 1;
						delim = strchr(p, ENV_DELIM);
					} else {
						scanning = false;
					}
				}
			} else if (error != NULL) {
				error(user);
			}
		}
	}
}

char** env_var_list_read(const char* env_var_value, size_t* count)
{
	struct env_var_list_scan_ctx ctx = {0};
	
	env_var_list_scan(env_var_value,
										env_var_list_count_entry,
										env_var_list_count_entry_error,
										&ctx.num_elems);
	
	if (ctx.num_elems) { 
		ctx.list = zalloc(ctx.num_elems * sizeof(char*));
	}
	
	if (ctx.list != NULL) {
		env_var_list_scan(env_var_value,
											env_var_list_insert_entry,
											NULL,
											&ctx);
	}

	if (count != NULL) {
		*count = ctx.num_elems;
	}

	return ctx.list;
}

struct test {
	uint8_t print_info;
} static g_test_params = {
	false
};

void test_env_var(char* str, size_t expected_count, bool should_null)
{
	if (g_test_params.print_info) {
		printf("Testing %s. Expecting %s with a count of %lu\n",
					 str,
					 should_null ? "failure" : "success",
					 expected_count);
	}
	
	size_t count = 0;
	char** list = env_var_list_read(str, &count);

	if (should_null) {
		assert(list == NULL);
		assert(count == 0);

		if (g_test_params.print_info) {
			printf("env_var_list_read for %s returned NULL\n", str);
		}
	} else {
		assert(count == expected_count);
		assert(list != NULL);

		for (size_t i = 0; i < count; ++i) {
			printf("[%lu]: %s\n", i, list[i]);
		}

		for (size_t i = 0; i < count; ++i) {
			assert(list[i] != NULL);
			free(list[i]);
		}

		free(list);
	}
}

void test_env_parse()
{
	test_env_var("BLANK=::", 0, 1);
	test_env_var("VALID=this:is:a:set:of:strings", 6, 0);
	test_env_var("MALFORMED=this::is:a::bad:string", 0, 1);
}

void test_env_read()
{
	getenv
}

int main()
{
	test_env_parse();
	
	gpu_test_matrix_vec_mul();
	
	gpu_test();
	
	CUDA_RUNTIME_FN(cudaDeviceSynchronize());
	
	return 0;
}
