#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <assert.h>

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                                                        __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
  do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
        __FILE__, __LINE__, #apiFuncCall, _status);                    \
      exit(-1);                                                              \
    }                                                                          \
  } while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
  do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
        __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
      exit(-1);                                                              \
    }                                                                          \
  } while (0)

typedef char cuptiAttrString_t[256];
typedef char cuptiAttrLargeString_t[1024];

/*
 * CUPTI_EVENT_CATEGORY_INSTRUCTION = 0
 An instruction related event. 
 CUPTI_EVENT_CATEGORY_MEMORY = 1
 A memory related event. 
 CUPTI_EVENT_CATEGORY_CACHE = 2
 A cache related event. 
 CUPTI_EVENT_CATEGORY_PROFILE_TRIGGER = 3
 A profile-trigger event. 
 CUPTI_EVENT_CATEGORY_SYSTEM = 4
 */

static const char* g_categories[5] = {
        "CUPTI_EVENT_CATEGORY_INSTRUCTION",
        "CUPTI_EVENT_CATEGORY_MEMORY",
        "CUPTI_EVENT_CATEGORY_CACHE",
        "CUPTI_EVENT_CATEGORY_PROFILE_TRIGGER",
        "CUPTI_EVENT_CATEGORY_SYSTEM"
};

static void cuptiInfo(CUdevice device) {
        uint32_t numDomains = 0;

        CUPTI_CALL(cuptiDeviceGetNumEventDomains(device, &numDomains));
        CUpti_EventDomainID* domainIDs = (CUpti_EventDomainID*) malloc(sizeof(CUpti_EventDomainID) * numDomains);

        assert(domainIDs != NULL);

        size_t szNumDomains = (size_t)(numDomains) * sizeof(CUpti_EventDomainID);
        CUPTI_CALL(cuptiDeviceEnumEventDomains(device, &szNumDomains, domainIDs));
        szNumDomains /= sizeof(CUpti_EventDomainID);

        printf("Domain Count: %lu\n", szNumDomains);

        for (size_t i = 0; i < szNumDomains; ++i) {
                CUpti_EventDomainID id = domainIDs[i];
                uint32_t numEvents = 0;

                CUPTI_CALL(cuptiEventDomainGetNumEvents(id, &numEvents));
                CUpti_EventID* eventIDs = (CUpti_EventID*) malloc(sizeof(CUpti_EventID) * numEvents);
                assert(eventIDs != NULL);

                size_t szNumEvents = ((size_t)numEvents) * sizeof(CUpti_EventID);
                CUPTI_CALL(cuptiEventDomainEnumEvents(id, &szNumEvents, eventIDs));
                szNumEvents /= sizeof(CUpti_EventID);

                cuptiAttrString_t dname = {0};
                size_t dname_len = sizeof(dname) - 1;
                CUPTI_CALL(cuptiEventDomainGetAttribute(id, CUPTI_EVENT_DOMAIN_ATTR_NAME, &dname_len, dname));

                printf("ID: %u. Domain Name: %s. Count: %lu\n", id, dname, szNumEvents);

                for (size_t j = 0; j < szNumEvents; ++j) {
                        CUpti_EventID eid = eventIDs[j];

                        cuptiAttrString_t name = {0};
                        size_t len = sizeof(name) - 1;
                        CUPTI_CALL(cuptiEventGetAttribute(eid, CUPTI_EVENT_ATTR_NAME, &len, name));

                        cuptiAttrLargeString_t desc = {0};
                        len = sizeof(desc) - 1;
                        CUPTI_CALL(cuptiEventGetAttribute(eid, CUPTI_EVENT_ATTR_LONG_DESCRIPTION, &len, desc));

                        CUpti_EventCategory cat;;
                        len = sizeof(cat);
                        CUPTI_CALL(cuptiEventGetAttribute(eid, CUPTI_EVENT_ATTR_CATEGORY, &len, &cat));

                        assert(0 <= (int)cat && (int)cat < 5);

                        printf("\n\tName: %s\n\t\tID: %i\n\t\tDescription: %s\n\t\tCategory: %s\n\n",
                                                 name,
                                                 eid,
                                                 desc,
                                                 g_categories[(int)cat]
                                );
                }

                free(eventIDs);
        }

        free(domainIDs);
}

int main(int argc, char** argv) {

        CUdevice device;
  char deviceName[128] = {0};
  int deviceNum = 0, devCount = 0;

        DRIVER_API_CALL(cuInit(0));

  RUNTIME_API_CALL(cudaGetDeviceCount(&devCount));
  for (deviceNum=0; deviceNum<devCount; deviceNum++) {
    DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
    DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, device));
    printf("Device Name: %s\n", deviceName);

    RUNTIME_API_CALL(cudaSetDevice(deviceNum));

                cuptiInfo(device);

    RUNTIME_API_CALL(cudaDeviceSynchronize());
    RUNTIME_API_CALL(cudaDeviceReset());

//              CUPTI_CALL(cuptiActivityFlushAll(0));
        }

        return 0;
}
