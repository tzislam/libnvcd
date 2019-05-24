#include "env_var.h"
#include "commondef.h"
#include "util.h"
#include <ctype.h>

C_LINKAGE_START

//
// Internal API
//

typedef int (*env_var_list_scan_fn_t)(const char* entry, size_t entry_len, void* user);

typedef void (*env_var_list_scan_error_fn_t)(void* user);

struct env_var_list_scan_ctx {
	char** list;
	size_t index;
	size_t num_elems;
};

static const char* env_var_list_start(const char* list) {
	const char* p = list;

	while (*p && *p != '=') {
		p++;
	}

	ASSERT(*p == '=');

	const char* ret = p + 1;

	if (!isalpha(*ret)) {
		printf("ERROR: %s must begin with a letter.\n", ret);
		ret = NULL;
	}

	return ret;
}

static const char* env_var_list_scan_entry(const char* p, size_t* p_count) {
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

static int env_var_list_count_entry(const char* entry, size_t entry_len, void* user) {
	ASSERT(user != NULL);
	size_t* count = (size_t*) user;
	*count = *count + 1;
	return 1;
}

static void env_var_list_count_entry_error(void* user) {
	ASSERT(user != NULL);

	size_t* count = (size_t*) user;
	*count = 0;
}

static int env_var_list_insert_entry(const char* entry, size_t entry_len, void* user) {
	ASSERT(user != NULL);
	struct env_var_list_scan_ctx* ctx = (struct env_var_list_scan_ctx*) user;
	
	char* str = zalloc((entry_len + 1) * sizeof(char));

	ASSERT(ctx->index < ctx->num_elems);
	
	if (str != NULL) {
		strncpy(str, entry, entry_len);

		ctx->list[ctx->index] = str;
		ctx->index++;
	}

	return str != NULL;
}

static void env_var_list_scan(const char* var,
											 env_var_list_scan_fn_t callback,
											 env_var_list_scan_error_fn_t error,
											 void* user) {
	const char* p = var;

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

//
// Public API
//

char** env_var_list_read(const char* env_var_value, size_t* count) {
	struct env_var_list_scan_ctx ctx = { 0 };
	
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

C_LINKAGE_END
