#include <cuda_runtime_api.h>

#define NVCD_HEADER_IMPL
#include <nvcd/nvcd.cuh>
#undef NVCD_HEADER_IMPL

#include <dlfcn.h>

static bool g_enabled = false;

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

NVCD_EXPORT void libnvcd_begin(const char* region_name) {
  strncpy(g_region_buffer, region_name, 255);
  g_enabled = true;
}

NVCD_EXPORT void libnvcd_end() {
  g_enabled = false;
  nvcd_run_info::num_runs = 0;
}

C_LINKAGE_END
