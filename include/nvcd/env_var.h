#ifndef __ENV_VAR_H__
#define __ENV_VAR_H__

#include "nvcd/commondef.h"

C_LINKAGE_START

#ifndef ENV_PREFIX
#define ENV_PREFIX "BENCH_"
#endif

#define ENV_EVENTS ENV_PREFIX "EVENTS"

#define ENV_METRICS ENV_PREFIX "METRICS"

#define ENV_DELIM ','
#define ENV_ALL_EVENTS "ALL"

char** env_var_list_read(const char* env_var_value, size_t* count);



C_LINKAGE_END

#endif //__ENV_VAR_H__
