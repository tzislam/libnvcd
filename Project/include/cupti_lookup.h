#ifndef __CUPTI_LOOKUP_H__
#define __CUPTI_LOOKUP_H__

#include "commondef.h"

C_LINKAGE_START

#define NUM_CUPTI_EVENTS_2X_DOMAIN_A 37
#define NUM_CUPTI_EVENTS_2X NUM_CUPTI_EVENTS_2X_DOMAIN_A

extern const char* g_cupti_events_2x[NUM_CUPTI_EVENTS_2X];




#define NUM_CUPTI_METRICS_3X 127

extern const char* g_cupti_metrics_3x[NUM_CUPTI_METRICS_3X];

C_LINKAGE_END
#endif //__CUPTI_LOOKUP_H__
