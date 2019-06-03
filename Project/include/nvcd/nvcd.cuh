#ifndef __NVCD_CUH__
#define __NVCD_CUH__

#include <nvcd/commondef.h>
#include <nvcd/util.h>
#include <nvcd/list.h>
#include <nvcd/env_var.h>
#include <nvcd/cupti_lookup.h>

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <set>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <ctype.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>

#include <pthread.h>

#include <cupti.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __CUDACC__
#ifndef PRIu64
#define PRIu64 "lu"
#endif

#ifndef PRId64
#define PRId64 "ld"
#endif

#ifndef PRIx64
#define PRIx64 "lx"
#endif

#ifndef PRIu32
#define PRIu32 "u"
#endif

#ifndef PRId32
#define PRId32 "i"
#endif

#ifndef PRIx32
#define PRIx32 "x"
#endif

#endif // __CUDACC__

#define EXTC extern "C"
#define DEV __device__
#define HOST __host__
#define GLOBAL __global__

#define NVCD_DEV_EXPORT static inline DEV
#define NVCD_CUDA_EXPORT static inline HOST
#define NVCD_GLOBAL_EXPORT static GLOBAL

#define DEV_PRINT_PTR(v) printf("&(%s) = %p, %s = %p\n", #v, &v, #v, v)

#define NVCD_KERNEL_EXEC(kname, dim3_grid, dim3_block, ...)       \
  do {                                                            \
    while (!nvcd_host_finished()) {                               \
      kname<<<dim3_grid, dim3_block>>>(__VA_ARGS__);              \
      CUDA_RUNTIME_FN(cudaDeviceSynchronize());                   \
    }                                                             \
  } while (0)

#define NVCD_KERNEL_EXEC_v2(kname, num_blocks, threads_per_block, ...)  \
  do {                                                                  \
    cupti_event_data_begin(&g_event_data);                              \
    while (!nvcd_host_finished()) {                                     \
      kname<<<num_blocks, threads_per_block>>>(__VA_ARGS__);            \
      CUDA_RUNTIME_FN(cudaDeviceSynchronize());                         \
    }                                                                   \
    cupti_event_data_end(&g_event_data);                                \
    NVCD_KERNEL_EXEC_METRICS(&g_event_data,                             \
                             kname,                                     \
                             num_blocks,                                \
                             threads_per_block,                         \
                             __VA_ARGS__);                              \
  } while (0)

#define NVCD_KERNEL_EXEC_METRICS(p_event_data, kname, num_blocks, threads_per_block, ...) \
  do {                                                                  \
    cupti_event_data_t* __e = (p_event_data);                           \
                                                                        \
    ASSERT(__e->is_root == true);                                       \
    ASSERT(__e->initialized == true);                                   \
    ASSERT(__e->metric_data != NULL);                                   \
    ASSERT(__e->metric_data->initialized == true);                      \
                                                                        \
    for (uint32_t i = 0; i < __e->metric_data->num_metrics; ++i) {      \
      cupti_event_data_begin(&__e->metric_data->event_data[i]);         \
      while (!cupti_event_data_callback_finished(&__e->metric_data->event_data[i])) { \
        kname<<<num_blocks, threads_per_block>>>(__VA_ARGS__);          \
        CUDA_RUNTIME_FN(cudaDeviceSynchronize());                       \
      }                                                                 \
                                                                        \
      cupti_event_data_end(&__e->metric_data->event_data[i]);           \
    }                                                                   \
  } while (0)                                                           

typedef struct nvcd {
  CUdevice* devices;
  CUcontext* contexts;

  char** device_names;
  
  int num_devices;
  
  bool32_t initialized;
} nvcd_t;

namespace detail {
  DEV clock64_t* dev_tstart = nullptr;
  DEV clock64_t* dev_ttime = nullptr;
  DEV int* dev_num_iter = nullptr;
  DEV uint* dev_smids = nullptr;
}

extern "C" {
  extern nvcd_t g_nvcd;
  
  extern cupti_event_data_t g_event_data;
  
  extern size_t dev_tbuf_size;
  extern size_t dev_num_iter_size;
  extern size_t dev_smids_size;

  extern void* d_dev_tstart;
  extern void* d_dev_ttime;
  extern void* d_dev_num_iter;
  extern void* d_dev_smids;

  extern volatile bool test_imbalance_detect;

  NVCD_CUDA_EXPORT void nvcd_device_get_ttime(clock64_t* out);

  NVCD_CUDA_EXPORT void nvcd_device_get_smids(unsigned* out);
}

//
// Stats
//

struct block {
  int thread;
  clock64_t time;
};

struct kernel_invoke_data {
  std::vector<block> load_minor;
  std::vector<block> load_major;
  
  std::vector<clock64_t> times;
  std::vector<uint32_t> smids;
  
  double time_stddev;
  double time_mean;

  size_t num_threads;
  
  kernel_invoke_data(size_t num_threads_)
    : times(num_threads_, 0),
      smids(num_threads_, 0),
      time_stddev(0.0),
      num_threads(num_threads_)
  {}

  ~kernel_invoke_data()
  {}

  void fill_outliers(double bounds,
                     double q1,
                     double q3,
                     const std::vector<block>& in,
                     std::vector<block>& out) {
    
    clock64_t max = static_cast<clock64_t>(q3) + static_cast<clock64_t>(bounds);
    
    printf("{q1, q3, bounds, max} = {%f, %f, %f, %" PRId64 "}\n",
           q1, q3, bounds, max);

    for (const block& b: in) {
      if (b.time > max) {
        out.push_back(b);
      }
    }
  }

  void print_blockv(const char* name, const std::vector<block>& v) {
    printf("=====%s=====\n", name);
    
    for (size_t i = 0; i < v.size(); ++i) {
      printf("[%lu] time = %" PRId64 " , thread = %" PRId32 "\n",
             i,
             v[i].time,
             v[i].thread);
    }
  }
  
  void write() {
    for (size_t i = 0; i < num_threads; ++i) {
      printf("[%lu] time = %" PRId64 ", smid = % " PRId32 "\n",
             i,
             times[i],
             smids[i]);
    }
    
    {
      std::vector<block> sorted;

      for (size_t i = 0; i < num_threads; ++i) {
        sorted.push_back(block{static_cast<int>(i), times[i]}); 
      }
      
      std::sort(sorted.begin(), sorted.end(), [](const block& a, const block& b) -> bool {
          return a.time < b.time;
        });

      print_blockv("sorted", sorted);

      size_t qlen = num_threads >> 2;

      double q1 = static_cast<double>(sorted[qlen].time)
        + static_cast<double>(sorted[qlen - 1].time);
      q1 = q1 * 0.5;

      double q3 = static_cast<double>(sorted[qlen * 3].time)
        + static_cast<double>(sorted[(qlen * 3) - 1].time);
      q3 = q3 * 0.5;

      double iqr = q3 - q1;

      double minorb = iqr * 1.5;
      fill_outliers(minorb, q1, q3, sorted, load_minor);

      double majorb = iqr * 3.0;
      fill_outliers(majorb, q1, q3, sorted, load_major);
    }

    print_blockv("load_minor", load_minor);
    print_blockv("load_major", load_major);
  }
};

template <typename T>
NVCD_CUDA_EXPORT void __buf_to_vec(std::vector<T>& vec, T* buf, uint32_t length) {
  vec.resize(length);
  memcpy(&vec[0], &buf[0], sizeof(buf[0]) * length);
  free(buf);
}

struct nvcd_device_info {
  struct entry {
    static constexpr uint32_t id_unset = static_cast<uint32_t>(-1);
    
    std::string name;
    uint32_t id;
    bool supported;

    bool blank() {
      return id == id_unset && supported == false && name == "";
    }
    
    entry()
      : entry("", id_unset, false) {
    }

    entry(const std::string& name_, bool supported_)
      : entry(name_, id_unset, supported)
        
    {}

    entry(const std::string& name_, uint32_t id_, bool supported_)
      : name(name_),
        id(id_),
        supported(supported_) {
    }
  };

  using id_list_type = std::vector<uint32_t>;
  using event_name_id_map_type = std::unordered_map<std::string, id_list_type>;
  
  struct metric_entry : public entry {
    event_name_id_map_type events;
    
    metric_entry(const std::string& name, bool supported, event_name_id_map_type events_)
      : entry(name, supported),
        events(std::move(events_))
      {}
  };
  
  using name_list_type = std::vector<entry>;
  using metric_list_type = std::vector<metric_entry>;
  
  using event_map_type = std::unordered_map<std::string,
                                            name_list_type>;

  using metric_map_type = std::unordered_map<std::string,
                                             metric_list_type>;
  
  using ptr_type = std::unique_ptr<nvcd_device_info>;
  
  event_map_type events;

  metric_map_type metrics;

  std::vector<std::string> device_names;
  
  event_name_id_map_type get_metric_event_names(const std::string& device,
                                                const std::string& metric_name,
                                                CUpti_MetricID metric_id) {    
    event_name_id_map_type event_names;
      
    uint32_t num_events;

    CUpti_EventID* event_ids = cupti_metric_get_event_ids(metric_id,
                                                          &num_events);

    for (uint32_t k = 0; k < num_events; ++k) {                            
      char* cname = cupti_event_get_name(event_ids[k]);
      
      std::string name(cname);
      free(cname);
      
      if (event_names.find(name) == event_names.end()) {
        event_names[name] = id_list_type();
      }
      
      event_names[name].push_back(event_ids[k]);
    }

    free(event_ids);

    return event_names;
  }

  bool event_supported(const std::string& device,
                       const std::string& name) {
    const auto& all_events = this->events.at(device);

    size_t l = 0;

    bool found = false;
    bool supported = true;

    entry ret;
          
    while (!found && l < all_events.size()) {
      found = all_events.at(l).name == name;
      
      if (found) {
        if (!all_events.at(l).supported) {
          supported = false;
        }
      }
                
      l++;
    }

    return supported;
  }
  
  bool all_events_supported(const std::string& device,
                            CUdevice device_handle,
                            const event_name_id_map_type& event_names) {
    bool all_supported = true;

    for (auto& event_name: event_names) {
      if (event_name.first != "event_name") {
        all_supported =
          event_supported(device, event_name.first);
      }

      if (!all_supported) {
        break;
      }
    }  
    
    return all_supported;
  }
  
  nvcd_device_info() {
    ASSERT(g_nvcd.initialized == true);

    // no need to free this list
    char** event_names = cupti_get_event_names();
    
    auto num_event_names = cupti_get_num_event_names();
    
    for (auto i = 0; i < g_nvcd.num_devices; ++i) {
      std::string device(g_nvcd.device_names[i]);
      CUdevice device_handle = g_nvcd.devices[i];
      
      device_names.push_back(device);

      // device events
      {
        events[device] = name_list_type();

        auto& list = events[device];

        for (decltype(num_event_names) j = 0; j < num_event_names; ++j) {
          std::string event(event_names[j]);
          
          CUpti_EventID event_id = static_cast<uint32_t>(-1);
          
          CUptiResult err = cuptiEventGetIdFromName(g_nvcd.devices[i],
                                                    event_names[j],
                                                    &event_id);
          bool supported = err == CUPTI_SUCCESS;
          
          entry e(event,
                  event_id,
                  supported);

          list.push_back(e);
        }
      }

      // device metrics
      {        
        uint32_t num_metrics = 0;
        
        CUpti_MetricID* metric_ids = cupti_metric_get_ids(g_nvcd.devices[i],
                                                          &num_metrics);
        metrics[device] = metric_list_type();

        auto& list = metrics[device];
        
        for (uint32_t j = 0; j < num_metrics; ++j) {
          char* cmetric_name = cupti_metric_get_name(metric_ids[j]);
          
          std::string metric_name(cmetric_name);
          
          free(cmetric_name);

          event_name_id_map_type event_names =
            get_metric_event_names(device,
                                   metric_name,
                                   metric_ids[j]);

          // the metric cannot be used if any of its dependent events
          // are unsupported
          bool all_supported = all_events_supported(device,
                                                    device_handle,
                                                    event_names);                                             
      
          metric_entry m(metric_name, all_supported, std::move(event_names));

          list.push_back(m);      
        }

        free(metric_ids);
      }
    }
  }
};

struct nvcd_run_info {
  std::vector<kernel_invoke_data> kernel_stats;
  std::vector<cupti_event_data_t> cupti_events;

  size_t num_runs;
  size_t curr_num_threads;
  
  nvcd_run_info()
    : num_runs(0),
      curr_num_threads(0) {}

  ~nvcd_run_info() {
    for (size_t i = 0; i < num_runs; ++i) {
      cupti_event_data_free(&cupti_events[i]);
    }
  }
  
  void update() {
    ASSERT(curr_num_threads != 0);
    
    {
        
      kernel_invoke_data d(curr_num_threads);

      nvcd_device_get_smids(&d.smids[0]);
      nvcd_device_get_ttime(&d.times[0]);
      
      kernel_stats.push_back(std::move(d));

      curr_num_threads = 0;
    }
    
    if (num_runs == cupti_events.size()) {
      if (cupti_events.size() == 0) {
        cupti_events.resize(8);

        for (size_t i = 0; i < cupti_events.size(); ++i) {
          cupti_event_data_set_null(&cupti_events[i]);
        }
      } else {
        size_t oldsz = cupti_events.size();
        cupti_events.resize(cupti_events.size() << 1);

        for (size_t i = oldsz; i < cupti_events.size(); ++i) {
          cupti_event_data_set_null(&cupti_events[i]);
        }
      }
    }

    memcpy(&cupti_events[num_runs],
           &g_event_data,
           sizeof(g_event_data));

    num_runs++;
  }

  void report() {
    for (size_t i = 0; i < num_runs; ++i) {
      printf("================================RUN %" PRIu64 "================================\n",
             i);
      
      kernel_stats[i].write();

      cupti_report_event_data(&cupti_events[i]);
    }
  }
};

extern std::unique_ptr<nvcd_run_info> g_run_info;

//
// Device functions
//

NVCD_DEV_EXPORT uint get_smid() {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
}

NVCD_DEV_EXPORT void nvcd_device_begin(int thread) {
  //  DEV_PRINT_PTR(detail::dev_tstart);
  detail::dev_tstart[thread] = clock64();
}

NVCD_DEV_EXPORT void nvcd_device_end(int thread) {
  detail::dev_ttime[thread] = clock64() - detail::dev_tstart[thread];
  detail::dev_smids[thread] = get_smid();

  // DEV_PRINT_PTR(detail::dev_ttime);
  // DEV_PRINT_PTR(detail::dev_smids);
}

NVCD_GLOBAL_EXPORT void nvcd_kernel_test() {
  int thread = blockIdx.x * blockDim.x + threadIdx.x;

  int num_threads = blockDim.x * gridDim.x;

  if (thread == 0) {
    
  }

  if (thread < num_threads) {
    nvcd_device_begin(thread);

    volatile int number = 0;

    for (int i = 0; i < detail::dev_num_iter[thread]; ++i) {
      number += i;
    }

    nvcd_device_end(thread);
  }
}


//
// CUDA template functions
//

template <class T>
static void cuda_safe_free(T*& ptr) {
  if (ptr != nullptr) {
    CUDA_RUNTIME_FN(cudaFree(static_cast<void*>(ptr)));
    ptr = nullptr;
  }
}


template <class T>
static void cuda_memcpy_host_to_dev(void* dst, std::vector<T> host) {
  size_t size = host.size() * sizeof(T);

  CUDA_RUNTIME_FN(cudaMemcpy(static_cast<void*>(dst),
                             static_cast<void*>(host.data()),
                             size,
                             cudaMemcpyHostToDevice));
}

template <class T>
NVCD_CUDA_EXPORT void* __cuda_zalloc_sym(size_t size, const T& sym, const char* ssym) {
  void* address_of_sym = nullptr;
  CUDA_RUNTIME_FN(cudaGetSymbolAddress(&address_of_sym, sym));
  ASSERT(address_of_sym != nullptr);

  CUDA_RUNTIME_FN(cudaMemset(address_of_sym, 0xFEED, sizeof(address_of_sym)));
  
  void* device_allocated_mem = nullptr;

  CUDA_RUNTIME_FN(cudaMalloc(&device_allocated_mem, size));
  ASSERT(device_allocated_mem != nullptr);

  CUDA_RUNTIME_FN(cudaMemset(device_allocated_mem, 0, size));

  CUDA_RUNTIME_FN(cudaMemcpy(address_of_sym,
                             &device_allocated_mem,
                             sizeof(device_allocated_mem),
                             cudaMemcpyHostToDevice));
  
  ASSERT(address_of_sym != nullptr);

  // sanity check
  {
    void* check = nullptr;
    
    CUDA_RUNTIME_FN(cudaMemcpy(&check,
                               address_of_sym,
                               sizeof(check),
                               cudaMemcpyDeviceToHost));

    printf("HOST-SIDE DEVICE ADDRESS FOR %s: %p. value: %p\n",
           ssym,
           address_of_sym,
           check);
    
    ASSERT(check == device_allocated_mem);
  }
  return device_allocated_mem;
}

#define cuda_zalloc_sym(sz, sym) __cuda_zalloc_sym(sz, sym, #sym)

//
// BASE API
//

extern "C" {
  NVCD_CUDA_EXPORT void nvcd_init_cuda(nvcd_t* nvcd) {
    if (!nvcd->initialized) {
      CUDA_DRIVER_FN(cuInit(0));
  
      CUDA_RUNTIME_FN(cudaGetDeviceCount(&nvcd->num_devices));

      nvcd->devices = (CUdevice*)zallocNN(sizeof(*(nvcd->devices)) *
                                          nvcd->num_devices);

      nvcd->contexts = (CUcontext*)zallocNN(sizeof(*(nvcd->contexts)) *
                                            nvcd->num_devices);

      nvcd->device_names = (char**)zallocNN(sizeof(*(nvcd->device_names)) *
                                            nvcd->num_devices);

      const size_t MAX_STRING_LENGTH = 128;
      
      for (int i = 0; i < nvcd->num_devices; ++i) {
        CUDA_DRIVER_FN(cuDeviceGet(&nvcd->devices[i], i));
        
        CUDA_DRIVER_FN(cuCtxCreate(&nvcd->contexts[i],
                                   0,
                                   nvcd->devices[i]));
        
        nvcd->device_names[i] = (char*) zallocNN(sizeof(nvcd->device_names[i][0]) *
                                                 MAX_STRING_LENGTH);
        
        CUDA_DRIVER_FN(cuDeviceGetName(&nvcd->device_names[i][0],
                                       MAX_STRING_LENGTH,
                                       nvcd->devices[i]));
      }

      nvcd->initialized = true;
    }
  }

  NVCD_CUDA_EXPORT void nvcd_device_free_mem() {
    cuda_safe_free(d_dev_tstart);
    cuda_safe_free(d_dev_ttime);
    cuda_safe_free(d_dev_num_iter);
    cuda_safe_free(d_dev_smids);
  }

  NVCD_CUDA_EXPORT void nvcd_device_init_mem(int num_threads) {
    {       
      dev_tbuf_size = sizeof(clock64_t) * static_cast<size_t>(num_threads);

      d_dev_tstart = cuda_zalloc_sym(dev_tbuf_size, detail::dev_tstart);
      d_dev_ttime = cuda_zalloc_sym(dev_tbuf_size, detail::dev_ttime);
    }

    {
      dev_smids_size = sizeof(uint) * static_cast<size_t>(num_threads);
      
      d_dev_smids = cuda_zalloc_sym(dev_smids_size, detail::dev_smids);
    }

    if (test_imbalance_detect) {
      dev_num_iter_size = sizeof(int) * static_cast<size_t>(num_threads);

      d_dev_num_iter = cuda_zalloc_sym(dev_num_iter_size, detail::dev_num_iter);

      std::vector<int> host_num_iter(num_threads, 0);

      int iter_min = 100;
      int iter_max = iter_min * 100;

      for (size_t i = 0; i < host_num_iter.size(); ++i) {
        srand(time(nullptr));

        if (i > host_num_iter.size() - 100) {
          iter_min = 1000;
          iter_max = iter_min * 100;
        }

        host_num_iter[i] = iter_min + (rand() % (iter_max - iter_min));
      }

      cuda_memcpy_host_to_dev<int>(d_dev_num_iter, std::move(host_num_iter));
    }
  }

  NVCD_CUDA_EXPORT void nvcd_device_get_ttime(clock64_t* out) {
    CUDA_RUNTIME_FN(cudaMemcpy(out,
                               d_dev_ttime,
                               dev_tbuf_size,
                               cudaMemcpyDeviceToHost));
  }

  NVCD_CUDA_EXPORT void nvcd_device_get_smids(unsigned* out) {
    CUDA_RUNTIME_FN(cudaMemcpy(out,
                               d_dev_smids,
                               dev_smids_size,
                               cudaMemcpyDeviceToHost));
  }

  NVCD_CUDA_EXPORT void nvcd_report() {
    ASSERT(g_run_info.get() != nullptr);
    
    g_run_info->report();
  }

  NVCD_CUDA_EXPORT void nvcd_init() {
    nvcd_init_cuda(&g_nvcd);

    if (!g_run_info) {
      g_run_info.reset(new nvcd_run_info());
    }
    
    printf("nvcd_init address: %p\n", nvcd_init);
    ASSERT(g_nvcd.initialized == true);

    g_event_data.is_root = true;
  }

  NVCD_CUDA_EXPORT void nvcd_host_begin(int num_cuda_threads) {  
    printf("nvcd_host_begin address: %p\n", nvcd_host_begin);

    ASSERT(g_nvcd.initialized == true);
    ASSERT(g_run_info.get() != nullptr);

    nvcd_device_init_mem(num_cuda_threads);

    g_run_info->curr_num_threads = static_cast<size_t>(num_cuda_threads);
    
    g_event_data.cuda_context = g_nvcd.contexts[0];
    g_event_data.cuda_device = g_nvcd.devices[0];
  
    cupti_event_data_init(&g_event_data);
  }

  NVCD_CUDA_EXPORT bool nvcd_host_finished() {
    return cupti_event_data_callback_finished(&g_event_data);
  }

  NVCD_CUDA_EXPORT void nvcd_host_end() {
    ASSERT(g_nvcd.initialized == true);
    
    cupti_event_data_calc_metrics(&g_event_data);

    g_run_info->update();
    
    nvcd_device_free_mem();

    cupti_event_data_set_null(&g_event_data);
  }
  
  NVCD_CUDA_EXPORT nvcd_device_info::ptr_type nvcd_host_get_device_info() {
    ASSERT(g_nvcd.initialized == true);
    nvcd_device_info::ptr_type ptr(new nvcd_device_info());
    return std::move(ptr);
  }

  NVCD_CUDA_EXPORT void nvcd_terminate() {
    cupti_event_data_free(&g_event_data);
    cupti_event_data_set_null(&g_event_data);
  
    cupti_name_map_free(); 

    g_run_info.reset();
 
    for (int i = 0; i < g_nvcd.num_devices; ++i) {
      ASSERT(g_nvcd.contexts[i] != NULL);
      safe_free_v(g_nvcd.device_names[i]);
      CUDA_DRIVER_FN(cuCtxDestroy(g_nvcd.contexts[i]));
    }

    safe_free_v(g_nvcd.device_names);
  }

  NVCD_CUDA_EXPORT void nvcd_kernel_test_call(int num_threads) {
    nvcd_host_begin(num_threads);
    
    int nblock = 4;
    int threads = num_threads / nblock;
    //    nvcd_kernel_test<<<nblock, threads>>>();

    NVCD_KERNEL_EXEC_v2(nvcd_kernel_test, nblock, threads);

    nvcd_host_end();
  }
}

//
// Define this in only one source file,
// and then include this header right after,
// and then undef it it
//
#ifdef NVCD_HEADER_IMPL

extern "C" {

  //
  // C++ doesn't support the nice
  // struct init syntax,
  // so these are commented out.
  //
  nvcd_t g_nvcd = {
    /*.devices =*/ NULL,
    /*.contexts =*/ NULL,
    /*.num_devices =*/ 0,
    /*.initialized =*/ false
  };

  cupti_event_data_t g_event_data = CUPTI_EVENT_DATA_NULL;

  size_t dev_tbuf_size = 0;
  size_t dev_num_iter_size = 0;
  size_t dev_smids_size = 0;

  void* d_dev_tstart = nullptr;
  void* d_dev_ttime = nullptr;
  void* d_dev_num_iter = nullptr;
  void* d_dev_smids = nullptr;

  volatile bool test_imbalance_detect = true;
}

std::unique_ptr<nvcd_run_info> g_run_info(nullptr);
  
#endif // NVCD_HEADER_IMPL

#endif // __NVCD_CUH__

