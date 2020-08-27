#include <cuda_runtime_api.h>

#define NVCD_HEADER_IMPL
#include <nvcd/nvcd.cuh>
#undef NVCD_HEADER_IMPL

#include <dlfcn.h>

#include <time.h>

#include <assert.h>

static bool g_enabled = false;

struct timer {
  struct timespec start;
  struct timespec end;
  double time;
}; 

#define NVCD_TIMEFLAGS_NONE 0
#define NVCD_TIMEFLAGS_REGION (1 << 0)
#define NVCD_TIMEFLAGS_KERNEL (1 << 1)
#define NVCD_TIMEFLAGS_RUN (1 << 2)

#define NVCD_TIMER_REGION 0
#define NVCD_TIMER_KERNEL 1
#define NVCD_TIMER_RUN 2
#define NVCD_NUM_TIMERS 3

struct hook_time_info {
  struct timer timers[3];
  uint32_t flags;
} static g_timer = {
  {
    { 0 },
    { 0 },
    { 0 }
  },
  0
};

static inline bool timer_set(uint32_t which) {
  return (g_timer.flags & which) == which;
}

static void timer_begin(uint8_t which) {
  assert(which < NVCD_NUM_TIMERS);
  struct timespec* start = &g_timer.timers[which].start;
  switch (which) {
  case NVCD_TIMER_REGION:
    if (timer_set(NVCD_TIMEFLAGS_REGION)) {
      clock_gettime(CLOCK_REALTIME, start);
    }
    break;
  default:
    break;
  }
}

static inline double timer_sec(struct timespec* t) {
  return (double)t->tv_sec + ((double)t->tv_nsec) * 1e-9;
}

static void timer_end(uint8_t which) {
  assert(which < NVCD_NUM_TIMERS);
  struct timespec* end = &g_timer.timers[which].end;
  bool isset = false;
  switch (which) {
  case NVCD_TIMER_REGION:
    isset = timer_set(NVCD_TIMEFLAGS_REGION);
    if (isset) {
      clock_gettime(CLOCK_REALTIME, end);
    }
    break;
  default:
    break;
  }

  if (isset) {
    g_timer.timers[which].time = 
      timer_sec(end) - timer_sec(&g_timer.timers[which].start);
  }
}

static void timer_report() {
  if (timer_set(NVCD_TIMEFLAGS_REGION)) {
    printf("[HOOK TIMER REGION]: %f\n", g_timer.timers[NVCD_TIMER_REGION].time);
  }
}

template <class TKernFunType,
	  class ...TArgs>
static inline cudaError_t nvcd_run_metrics2(const TKernFunType& kernel, 				     
					    TArgs... args) {
  cupti_event_data_t* __e = nvcd_get_events();                           
  
  ASSERT(__e->is_root == true);                                       
  ASSERT(__e->initialized == true);                                   
  ASSERT(__e->metric_data != NULL);                                   
  ASSERT(__e->metric_data->initialized == true);                      

  cudaError_t result = cudaSuccess;
  for (uint32_t i = 0; result == cudaSuccess && i < __e->metric_data->num_metrics; ++i) {      
    cupti_event_data_begin(&__e->metric_data->event_data[i]);         

    while (result == cudaSuccess && !cupti_event_data_callback_finished(&__e->metric_data->event_data[i])) {
      kernel(args...);                       
      CUDA_RUNTIME_FN(cudaDeviceSynchronize());                       
      g_run_info->run_kernel_count_inc();				
    }                                                                 
                                                                        
    cupti_event_data_end(&__e->metric_data->event_data[i]);
  }

  return result;
}


template <class TKernFunType, class ...TArgs>
static inline cudaError_t nvcd_run2(const TKernFunType& kernel, 
				    TArgs... args) {

  cudaError_t result = cudaSuccess;

  if (nvcd_has_events()) {
    cupti_event_data_begin(nvcd_get_events());  
    while (result == cudaSuccess && !nvcd_host_finished()) {                                     
      result = kernel(args...);                       
      CUDA_RUNTIME_FN(cudaDeviceSynchronize());                         
      g_run_info->run_kernel_count_inc();			
    }                                                                   
    cupti_event_data_end(nvcd_get_events());
  }

  if (result == cudaSuccess && nvcd_has_metrics()) {  
    result = nvcd_run_metrics2(kernel, args...);
  }

  return result;
}

C_LINKAGE_START

static char g_region_buffer[256] = {0};

typedef cudaError_t (*cudaLaunchKernel_fn_t)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);

static cudaLaunchKernel_fn_t real_cudaLaunchKernel = NULL;

NVCD_EXPORT __host__ cudaError_t cudaLaunchKernel(const void* func,
						  dim3 gridDim,
						  dim3 blockDim,
						  void** args,
						  size_t sharedMem,
						  cudaStream_t stream) {
  cudaError_t ret = cudaSuccess;
  if (real_cudaLaunchKernel == NULL) {
    real_cudaLaunchKernel = (cudaLaunchKernel_fn_t) dlsym(RTLD_NEXT, "cudaLaunchKernel");
  }
  if (g_enabled) {
    printf("[HOOK ON %s - %s]\n", __FUNC__, g_region_buffer);
    nvcd_host_begin(g_region_buffer, gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z);
    ret = nvcd_run2(real_cudaLaunchKernel, func, gridDim, blockDim, args, sharedMem, stream);
    nvcd_host_end();
  }
  else {
    printf("[HOOK OFF %s]\n", __FUNC__);
    ret = real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  }
  return ret;
}

NVCD_EXPORT void libnvcd_time(uint32_t flags) {
  g_timer.flags = flags;
}

NVCD_EXPORT void libnvcd_begin(const char* region_name) {
  strncpy(g_region_buffer, region_name, 255);
  g_enabled = true;
  timer_begin(NVCD_TIMER_REGION);
}

NVCD_EXPORT void libnvcd_end() {
  timer_end(NVCD_TIMER_REGION);
  g_enabled = false;
  nvcd_run_info::num_runs = 0;
  timer_report();
}

C_LINKAGE_END
