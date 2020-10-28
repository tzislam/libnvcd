#include <cuda_runtime_api.h>

#include <nvcd/commondef.h>
#include <nvcd/util.h>
#include <nvcd/env_var.h>
#include <nvcd/cupti_util.h>
#include <nvcd/nvcd.h>

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_set>
#include <limits>
#include <type_traits>
#include <sstream>
#include <iomanip>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <ctype.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#include <dirent.h>
#include <sys/stat.h>
#include <ftw.h>

#include <dlfcn.h>
#include <assert.h>
#include <string.h>
#include <cstdint>

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

#define __TONAME___DEV_EXPORT static inline DEV
#define __TONAME___CUDA_EXPORT static inline HOST
#define __TONAME___GLOBAL_EXPORT static GLOBAL

#define STREAM_HEX(bytes) "0x" << std::uppercase << std::setfill('0') << std::setw((bytes) << 1) << std::hex

#define DEV_PRINT_PTR(v) msg_verbosef("&(%s) = %p, %s = %p\n", #v, &v, #v, v)                                                           

using instance_vec_type = std::vector<uint64_t>;
using counter_map_type = std::unordered_map<CUpti_EventID, instance_vec_type>;

instance_vec_type operator - (const instance_vec_type& a, const instance_vec_type& b) {
  ASSERT(a.size() == b.size());
  // in case asserts are disabled (for whatever unlikely reason that may be...)
  size_t sz = std::min(a.size(), b.size());
  instance_vec_type diff(sz, 0);
  for (size_t i = 0; i < sz; ++i) {
    diff[i] = a[i] - b[i];
  }
  return diff;
} 

counter_map_type operator - (const counter_map_type& a, const counter_map_type& b) {
  counter_map_type diff;
  for (const auto& kv: a) {
    const auto& key = kv.first;
    const auto& value = kv.second;
    const instance_vec_type& avec = value;
    // b will be empty the first time this operator overload
    // is called, but this is just a better catch all solution.
    // sure, it's less efficient because b.empty() is simpler to check for.
    // the map is unordered though, so b.find() should have little overhead.
    if (b.find(key) != b.end()) {
      const instance_vec_type& bvec = b.at(key);
      diff[key] = avec - bvec;
    }
    else {
      diff[key] = avec;
    }
  }
  return diff;
}

struct hook_run_info {
  
  counter_map_type counters_start;
  counter_map_type counters_end;
  counter_map_type counters_diff;

  std::string region_name;
  
  size_t curr_num_threads;
  const char* func_name;
  uint32_t run_kernel_exec_count;

  static size_t num_runs;
  
  hook_run_info()
    : curr_num_threads(0),
      func_name(nullptr),
      run_kernel_exec_count(0) {
  }

  ~hook_run_info() {
  }

  void run_kernel_count_inc() {
    run_kernel_exec_count++;
  }
  
  void update(cupti_event_data_t* event_data) {
    ASSERT(curr_num_threads != 0);   

    curr_num_threads = 0;
    run_kernel_exec_count = 0;
   

    // we do this to compute the difference
    // from the previous run
    for (const auto& kv: counters_end) {
      counters_start[kv.first] = kv.second;      
    }
    
    cupti_event_data_enum_event_counters(event_data,
                                         this,
                                         hook_run_info::enum_event_counters);    
    
    counters_diff = counters_end - counters_start;

    num_runs++;
  }

  static bool enum_event_counters(cupti_enum_event_counter_iteration_t* it) {
    hook_run_info* run_info = static_cast<hook_run_info*>(it->user_param);
    if (run_info->counters_end[it->event].empty()) {
      run_info->counters_end[it->event].resize(it->num_instances, 0);
    }
    ASSERT(it->instance < it->num_instances);
    run_info->counters_end[it->event][it->instance] += it->value;
    return true;
  }
  
  void report(cupti_event_data_t* event_data) {
    ASSERT(num_runs > 0);
    
    msg_userf("================================ invocation %" PRIu64 " for \'%s\' ================================\n",
	      num_runs - 1,
	      region_name.c_str());

    std::stringstream ss;
    msg_verbosef("counters_diff size: %" PRIu64 "\n", counters_diff.size());
    for (const auto& kv : counters_diff) {
      const auto& key = kv.first;
      const auto& value = kv.second;
      ASSERT(!value.empty());
      char* event_name = cupti_event_get_name(key);
      ASSERT(event_name != nullptr);
      double avg = 0;
      uint64_t summation = 0;
      uint64_t maximum = 0; // the lowest possible count
      uint64_t minimum = std::numeric_limits<uint64_t>::max(); //something very large so that it changes
      uint64_t temp_var = 0;
      for (size_t index = 0; index < value.size(); ++index) {
	temp_var = value.at(index);
	summation += temp_var;
	avg += temp_var;
	maximum = (maximum < temp_var) ? temp_var : maximum;
	minimum = (minimum > temp_var) ? temp_var : minimum;
      }
      avg /= static_cast<double>(value.size());
      ss << "|COUNTER|" << region_name << ":" << event_name << ": SUM: " << summation << " AVG: " << avg << " MAX: " << maximum << " MIN: " << minimum << "\n";
      free(event_name);
    }
    
    msg_userf("%s", ss.str().c_str());
    
    cupti_report_event_data(event_data);
  }
};

size_t hook_run_info::num_runs = 0;

static hook_run_info* g_run_info = nullptr;

__TONAME___CUDA_EXPORT void __toname___report() {
  ASSERT(g_run_info != nullptr);    
  g_run_info->report(nvcd_get_events());
}

__TONAME___CUDA_EXPORT void __toname___init() {
  nvcd_init_cuda();

  if (g_run_info == nullptr) {
    g_run_info = new hook_run_info();
  }
        
  ASSERT(g_nvcd.initialized == true);
  ASSERT(g_run_info != nullptr);
}

__TONAME___CUDA_EXPORT void __toname___host_begin(const char* region_name, int num_cuda_threads) {     
  __toname___init();

  g_run_info->region_name = std::string(region_name);

  ASSERT(g_nvcd.initialized == true);
  ASSERT(g_run_info != nullptr);

  g_run_info->curr_num_threads = static_cast<size_t>(num_cuda_threads);

  nvcd_init_events(g_nvcd.devices[0],
                   g_nvcd.contexts[0]);
}

__TONAME___CUDA_EXPORT bool __toname___host_finished() {
  return cupti_event_data_callback_finished(nvcd_get_events());
}

__TONAME___CUDA_EXPORT void __toname___terminate();

__TONAME___CUDA_EXPORT void __toname___host_end() {
  ASSERT(g_nvcd.initialized == true);
    
  nvcd_calc_metrics();

  g_run_info->update(nvcd_get_events());

  __toname___report();   

  __toname___terminate();
}
 

__TONAME___CUDA_EXPORT void __toname___terminate() {
  nvcd_reset_event_data();
 
  for (int i = 0; i < g_nvcd.num_devices; ++i) {
    ASSERT(g_nvcd.contexts[i] != nullptr);
    safe_free_v(g_nvcd.device_names[i]);
            
    if (g_nvcd.contexts_ext[i] == false) {
      CUDA_DRIVER_FN(cuCtxDestroy(g_nvcd.contexts[i]));
    }
  }

  safe_free_v(g_nvcd.device_names);
  safe_free_v(g_nvcd.devices);
  safe_free_v(g_nvcd.contexts);

  g_nvcd.initialized = false;
}

#define NVCD_TIMEFLAGS_NONE 0
#define NVCD_TIMEFLAGS_REGION (1 << 2)
#define NVCD_TIMEFLAGS_KERNEL (1 << 1)
#define NVCD_TIMEFLAGS_RUN (1 << 0)

enum spec
  {
   f_000 = NVCD_TIMEFLAGS_NONE,
   f_00r = NVCD_TIMEFLAGS_RUN,
   f_0k0 = NVCD_TIMEFLAGS_KERNEL,
   f_0kr = NVCD_TIMEFLAGS_KERNEL | NVCD_TIMEFLAGS_RUN,
   f_r00 = NVCD_TIMEFLAGS_REGION,
   f_r0r = NVCD_TIMEFLAGS_REGION | NVCD_TIMEFLAGS_RUN,
   f_rk0 = NVCD_TIMEFLAGS_REGION | NVCD_TIMEFLAGS_KERNEL,
   f_rkr = NVCD_TIMEFLAGS_REGION | NVCD_TIMEFLAGS_KERNEL | NVCD_TIMEFLAGS_RUN       
  };

struct timeflags { 
private:
  uint32_t value;
  
public:
  timeflags(uint32_t value)    
    : value(check(value, f_000, f_rkr))
  {}

  uint32_t check(uint32_t x, uint32_t min, uint32_t max) {
    if (!(min <= x && x <= max)) {
      exit_msg(stdout,
	       EBAD_INPUT,
	       "[HOOKE ERROR] value = %" PRIu32 " is out of defined range [%" PRIu32 ", %" PRIu32 "].\n",
	       x,
	       min,
	       max);
    }
    return x;
  }

  operator uint32_t () const {
    return value;
  }
};


typedef std::unordered_map<uint32_t, std::vector<std::string>> time_output_map_type;

static time_output_map_type time_map
  {
   {
    f_00r,
    {
     "run"
    }
   },

   {
    f_0k0,
    {
     "kernel"
    }
   },

   {
    f_0kr,
    {
     "kernel",
     "run"
    }
   },

   {
    f_r00,
    {
     "region"
    }
   },

   {
    f_r0r,
    {
     "region",
     "run"
    }
   },

   {
    f_rk0,
    {
     "region",
     "kernel"
    }
   },

   {
    f_rkr,
    {
     "region",
     "kernel",
     "run"
    }
   }
  };

static bool g_enabled = false;

struct timeslice {
  struct timespec start;
  struct timespec end;
  double time;
  bool set;

  timeslice() { reset(); }

  timeslice& reset() {
    start.tv_sec = start.tv_nsec = 0;
    end.tv_sec = end.tv_nsec = 0;
    time = 0.0;
    set = false;
    return *this;
  }

  timeslice& go() {
    clock_gettime(CLOCK_REALTIME, &start);
    return *this;
  }

  double seconds(struct timespec* t) const { return (double)t->tv_sec + ((double)t->tv_nsec) * 1e-9; }

  timeslice& stop() {
    if (!set) {
      clock_gettime(CLOCK_REALTIME, &end);
      time = seconds(&end) - seconds(&start);
    }
    return *this;
  }

  operator double () const {
    return time;
  }
};


struct timetree {
  enum ttype
    {
     node,
     leaf
    };

  const ttype type;
  
  timetree(ttype type) : type(type) {}
  
  using ptr_type = std::unique_ptr<timetree>;
  
  virtual double value() const = 0;

  virtual std::string to_string(const std::string& region, timeflags flags, uint32_t depth) const {
    std::stringstream ret;
    std::string title{time_map.at(flags).at(depth)};
    std::string tabs = (depth != 0) ? std::string(depth, '\t') : "";
    ret << tabs << "[HOOK TIME " << ((title == "region") ? ("region " + region) : title) <<  "] " << value() << " seconds\n";
    return ret.str();
  }
};

struct timenode : public timetree {
  std::vector<timetree::ptr_type> children;

  timenode() : timetree(node) {}
  
  double value() const override {
    double ret = 0.0;
    for (const auto& child: children) {
      ret += child->value();
    }    
    return ret;
  }

  std::string to_string(const std::string& region, timeflags flags, uint32_t depth) const override {
    std::stringstream ret;
    ret << timetree::to_string(region, flags, depth);
    for (const auto& child: children) {
      ret << child->to_string(region, flags, depth + 1);
    }
    return ret.str();
  }
};

struct timeleaf : public timetree {
  timeslice v;

  timeleaf() : timetree(leaf) {}

  double value() const override {
    return static_cast<double>(v);
  }
};

struct hook_time_record {
  std::string region_name;
  std::vector<std::string> dumps;
};

static std::vector<hook_time_record> g_time_records;

using call_interval_type = int32_t;

static constexpr call_interval_type k_max_call_interval{10000};
static constexpr call_interval_type k_min_call_interval{0};
static constexpr call_interval_type k_unset_call_interval{-1};

struct kernel_interval_params {
  static call_interval_type interval;
  call_interval_type call_count;
  
  kernel_interval_params()
    : call_count(0) {

    ASSERT((interval == k_unset_call_interval) ||
	   (k_min_call_interval <= interval &&
	    interval <= k_max_call_interval));
    
    if (interval == k_unset_call_interval) {
      char* interval_str = getenv(ENV_SAMPLE);

      if (interval_str != nullptr) {
	bool ok = false;
	// we restrict ourselves currently to a single value
	char* end_ptr = nullptr;
	call_interval_type ci = strtol(interval_str, &end_ptr, 10);
	  
	ok =
	  C_ASSERT(k_min_call_interval <= ci) &&
	  C_ASSERT(ci <= k_max_call_interval) &&
	  // ensures the entire string is a valid base 10 integer
	  C_ASSERT(end_ptr[0] == '\0' &&
		   interval_str[0] != '\0');
	  
	if (ok) {
	  interval = ci;
	}
	else {
	  interval = k_min_call_interval;
	}
      }
      else {
	interval = k_min_call_interval;
      }

      printf("[HOOK CALL INTERVAL = %" PRId32"]\n", interval);
    }
  }
};

call_interval_type kernel_interval_params::interval{k_unset_call_interval};

static std::unordered_map<uintptr_t, kernel_interval_params>  g_call_counts;

namespace {
  struct push {
    timetree::ptr_type& src;

    void to(timetree::ptr_type& dst) {
      ASSERT(dst->type == timetree::node);
      dynamic_cast<timenode*>(dst.get())->children.push_back(std::move(src));
    }
  };
  
  struct stop {
    timetree::ptr_type& src;

    stop(timetree::ptr_type& src)
      : src(src)
    {
      ASSERT(src->type == timetree::leaf);
      dynamic_cast<timeleaf*>(src.get())->v.stop();
    }
    
    void then_push_to(timetree::ptr_type& dst) {
      push { src }.to(dst);
    }
  };

  template <class T>
  struct set_traits {
    timetree::ptr_type& v;
    set_traits(timetree::ptr_type& v) : v(v) { v.reset(new T()); }   
  };

  template <class T>
  struct set_to : public set_traits<T> {
    set_to(timetree::ptr_type& v) : set_traits<T>(v) {}
    
    void then_start() {
      exit_msg(stdout,
	       EBAD_PATH,
	       "%s\n",
	       "[HOOK ERROR] then_start() reached with invalid type.\n");      
    }
  };

  template <>
  struct set_to<timeleaf> : public set_traits<timeleaf> {
    set_to(timetree::ptr_type& v) : set_traits<timeleaf>(v) {}
    
    void then_start() {
      dynamic_cast<timeleaf*>(v.get())->v.go();
    }
  };

  const std::vector<timetree::ptr_type>& get_children(const timetree::ptr_type& parent) {
    return dynamic_cast<timenode*>(parent.get())->children;
  }

  struct add {
    const std::string& dump;

    void to(const std::string& region_name) {
      bool found = false;
      for (auto& entry: g_time_records) {
        found = entry.region_name == region_name;
        if (found) {
          entry.dumps.push_back(dump);
          break;
        }
      }
      if (!found) {
        g_time_records.push_back(hook_time_record {region_name, {dump}});
      }
    }
  };

  struct call_for {
    uintptr_t symaddr;

    call_for(const void* func)
      : symaddr{reinterpret_cast<uintptr_t>(func)}{    
    }
  
    bool is_ready() {
      bool ready = (g_call_counts[symaddr].call_count % kernel_interval_params::interval) == 0;
      g_call_counts[symaddr].call_count++;
      return ready;
    }    
  };
}

struct hook_time_info {
  timetree::ptr_type root;
  
  timetree::ptr_type region;
  timetree::ptr_type kernel;
  timetree::ptr_type run;

  std::string region_name;
  
  timeflags flags;

  hook_time_info ()
    : root(nullptr),
      region(nullptr),
      kernel(nullptr),
      run(nullptr),
      flags(0)
  {
    set_to<timenode>{root};
  }

  uint32_t f32() const { return static_cast<uint32_t>(flags); }
  
  bool test(uint32_t f) const { return (f32() & f) == f; }
  
  bool has_region() const { return test(NVCD_TIMEFLAGS_REGION); }
  bool has_kernel() const { return test(NVCD_TIMEFLAGS_KERNEL); }
  bool has_run() const { return test(NVCD_TIMEFLAGS_RUN); }
  
  void begin_region(const char* region_name) {
    this->region_name = std::string(region_name);
    if (has_region()) {
      switch (f32()) {
      case f_rkr:
      case f_rk0:
      case f_r0r:
	set_to<timenode>{region};
	break;
      case f_r00:
	set_to<timeleaf>{region}.then_start();
	break;
      }
    }
  }

  void begin_kernel() {
    if (has_kernel()) {
      switch (f32()) {
      case f_rkr:
      case f_0kr:
	set_to<timenode>{kernel};
	break;
	
      case f_rk0:
      case f_0k0:
	set_to<timeleaf>{kernel}.then_start();
	break;
      }
    }    
  }
  
  void begin_run() {
    if (has_run()) {
      set_to<timeleaf>{run}.then_start();
    }
  }

  void end_run() {
    if (has_run()) {

      switch (f32()) {
      case f_rkr:
      case f_0kr:
	stop{run}.then_push_to(kernel);
	break;
      case f_r0r:
	stop{run}.then_push_to(region);
	break;
      case f_00r:
	stop{run}.then_push_to(root);
	break;
      }
    }
  }

  void end_kernel() {
    if (has_kernel()) {
      switch (f32()) {
      case f_rk0:
	stop{kernel}.then_push_to(region);	
	break;

      case f_0k0:
	stop{kernel}.then_push_to(root);
	break;
	
      case f_0kr:
	push{kernel}.to(root);
	break;
	
      case f_rkr:
	push{kernel}.to(region);
	break;
      }
    }
  }

  void end_region() {
    if (has_region()) {
      switch (f32()) {
      case f_r00:
	stop{region}.then_push_to(root);
	
	break;
      case f_r0r:
      case f_rk0:
      case f_rkr:
	push{region}.to(root);
	break;
      }
    }
  }

  void record() {
    ASSERT(!region_name.empty());
    std::stringstream ss;
    for (const auto& child_ptr: get_children(root)) {
      ss << child_ptr->to_string(region_name, flags, 0);
    }
    add{ss.str()}.to(region_name);
  }
};

static std::unique_ptr<hook_time_info> g_timer{nullptr};

static void reset_timer() {
  ASSERT(!g_enabled);
  if (!g_enabled) {
    if (g_timer) {
      timeflags tmp(g_timer->flags);
      g_timer.reset(new hook_time_info());
      g_timer->flags = tmp;
    }
  }
}

template <class TKernFunType,
	  class ...TArgs>
static inline cudaError_t nvcd_run_metrics2(const TKernFunType& kernel, 				     
					    TArgs... args) {
  cupti_event_data_t* __e = nvcd_get_events();                           
  
  ASSERT(__e->is_root == true);                                       
  ASSERT(__e->initialized == true);                                   
  ASSERT(__e->metric_data != nullptr);                                   
  ASSERT(__e->metric_data->initialized == true);                      

  cudaError_t result = cudaSuccess;
  for (uint32_t i = 0; result == cudaSuccess && i < __e->metric_data->num_metrics; ++i) {      
    cupti_event_data_begin(&__e->metric_data->event_data[i]);         

    while (result == cudaSuccess && !cupti_event_data_callback_finished(&__e->metric_data->event_data[i])) {
      if (g_timer) g_timer->begin_run();
      kernel(args...);                       
      CUDA_RUNTIME_FN(cudaDeviceSynchronize());
      if (g_timer) g_timer->end_run();
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
    while (result == cudaSuccess && !__toname___host_finished()) {
      if (g_timer) g_timer->begin_run();
      result = kernel(args...);                       
      CUDA_RUNTIME_FN(cudaDeviceSynchronize());
      if (g_timer) g_timer->end_run();
      g_run_info->run_kernel_count_inc();			
    }                                                                   
    cupti_event_data_end(nvcd_get_events());
  }

  if (result == cudaSuccess && nvcd_has_metrics()) {  
    result = nvcd_run_metrics2(kernel, args...);
  }

  return result;
}

typedef cudaError_t (*cudaLaunchKernel_fn_t)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);

class hook_driver {
public:
  using host_thread_id_type = pthread_t;

  hook_driver(host_thread_id_type htid)
    : m_run_info(new hook_run_info()),
      m_host_thread_id(htid),
      m_hook_enabled(false)
  {}
  
  void event_trace_begin(int num_cuda_threads, cudaStream_t stream) {
    ASSERT(!m_region_name.empty());
    ASSERT(num_cuda_threads > 0);
    
    m_run_info->region_name = m_region_name;
    m_run_info->curr_num_threads = static_cast<size_t>(num_cuda_threads);
    
    m_per_trace.begin(stream);
  }

  template <class ...TArgs>
  cudaError_t run(cudaLaunchKernel_fn_t real_cudaLaunchKernel, TArgs... args) {
    cudaError_t result = cudaSuccess;

    auto run_trace_loop_for = [&](cupti_event_data_t* e) {
      cupti_event_data_begin(e);
      while (result == cudaSuccess && !cupti_event_data_callback_finished(e)) {
        result = real_cudaLaunchKernel(args...);
        CUDA_RUNTIME_FN(cudaDeviceSynchronize());
        m_run_info->run_kernel_count_inc();
      }
      cupti_event_data_end(e);
    };
    
    if (event_data()->has_events) {
      run_trace_loop_for(event_data());
    }

    if (result == cudaSuccess && event_data()->has_metrics) {  
      ASSERT(event_data()->is_root == true);                                       
      ASSERT(event_data()->initialized == true);                                   
      ASSERT(event_data()->metric_data != nullptr);                                   
      ASSERT(event_data()->metric_data->initialized == true);                      

      cupti_event_data_t* metric_event_buffer =
        & (event_data()->metric_data->event_data[0]);

      for (uint32_t i = 0;
           result == cudaSuccess &&
             i < event_data()->metric_data->num_metrics;
           ++i) {
        run_trace_loop_for(&metric_event_buffer[i]);
      }
    }

    return result;
  }
  
  void event_trace_end() {
    m_per_trace.calc_metrics();
    
    m_run_info->update(event_data());
    m_run_info->report(event_data());
    
    m_per_trace.end();
  }
  
  void region_begin(std::string region_name) {
    m_region_name = std::move(region_name);
  }

  void region_end() {
    
  }

  
private:
    struct per_trace {
      cupti_event_data_t m_cupti_event_data;
      CUcontext m_cu_context;
      CUdevice m_cu_device;
      cudaStream_t m_cuda_stream;
      int m_device;
    
      bool m_cu_context_is_creat;
    
      per_trace() {
        reset();
      }

      void reset() {
        cupti_event_data_set_null(&m_cupti_event_data);
        m_cu_context = nullptr;
        m_cu_device = -1;
        m_cuda_stream = nullptr;
        m_device = -1;
        m_cu_context_is_creat = false;
      }

      void begin(cudaStream_t stream) {
        m_cuda_stream = stream;
      
        CUDA_RUNTIME_FN(cudaGetDevice(&m_device));
    
        CUDA_DRIVER_FN(cuDeviceGet(&m_cu_device, m_device));

        CUDA_DRIVER_FN(cuCtxGetCurrent(&m_cu_context));
        
        if (m_cu_context == nullptr) {
          CUDA_DRIVER_FN(cuCtxCreate(&m_cu_context,
                                     0,
                                     m_cu_device));

          m_cu_context_is_creat = true;
        }

        m_cupti_event_data.cuda_context = m_cu_context;
        m_cupti_event_data.cuda_device = m_cu_device;
        m_cupti_event_data.is_root = true;

        cupti_event_data_init(&m_cupti_event_data);
      }

      void calc_metrics() {
        if (m_cupti_event_data.has_metrics) {
          cupti_event_data_calc_metrics(&m_cupti_event_data);
        }
      }

      void end() {
        cupti_event_data_free(&m_cupti_event_data);
        if (m_cu_context_is_creat) {
          ASSERT(m_cu_context != nullptr);
          CUDA_DRIVER_FN(cuCtxDestroy(m_cu_context));
        }
        reset();
      }
    };
  
  cupti_event_data_t* event_data() { return &m_per_trace.m_cupti_event_data; }
  
  per_trace m_per_trace;
  
  std::unique_ptr<hook_run_info> m_run_info;

  std::string m_region_name;
  
  host_thread_id_type m_host_thread_id;
  
  bool m_hook_enabled;
};

class hook_driver_manager {
public:
  hook_driver* driver_for(hook_driver::host_thread_id_type thread) {
    if (!m_hooks[thread]) {
      m_hooks[thread].reset(new hook_driver(thread));
    }
    return m_hooks.at(thread).get();
  }
  
private:
  std::unordered_map<hook_driver::host_thread_id_type,
                     std::unique_ptr<hook_driver>> m_hooks;
};

static std::unique_ptr<hook_driver_manager> g_drvman(new hook_driver_manager());

C_LINKAGE_START

static char g_region_buffer[256] = {0};

static cudaLaunchKernel_fn_t real_cudaLaunchKernel = nullptr;

void print_func(const void* func) {
  const char* f = static_cast<const char*>(func);
  printf("[HOOK INFO - func: string = %s, address %p" PRIx64 "]\n", f, func);
}

NVCD_EXPORT __host__ cudaError_t cudaLaunchKernel(const void* func,
						  dim3 gridDim,
						  dim3 blockDim,
						  void** args,
						  size_t sharedMem,
						  cudaStream_t stream) {
  cudaError_t ret = cudaSuccess;
  
  if (real_cudaLaunchKernel == nullptr) {
    real_cudaLaunchKernel = (cudaLaunchKernel_fn_t) dlsym(RTLD_NEXT, "cudaLaunchKernel");
  }
  if (g_enabled) {
    if (call_for(func).is_ready()) {
      printf("[HOOK ON %s - %s; symbol = %p]\n", __FUNC__, g_region_buffer, func);
      if (g_timer) {
	g_timer->begin_kernel();
      }
      __toname___host_begin(g_region_buffer, gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z);
      ret = nvcd_run2(real_cudaLaunchKernel, func, gridDim, blockDim, args, sharedMem, stream);
      __toname___host_end();
      if (g_timer) {
	g_timer->end_kernel();
      }
    }
  }
  else {
    printf("[HOOK OFF %s]\n", __FUNC__);
    ret = real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  }
  //  print_func(func);
  return ret;
}

NVCD_EXPORT void libnvcd_time(uint32_t flags) {
  // We absolutely don't want to mess with the timer state
  // if a region has been enabled.
  ASSERT(!g_enabled);
  if (!g_enabled) {
    // user disabled timer, so we'll refrain from
    // continuing to record.
    if (flags == 0) {
      g_timer.reset(nullptr);
    } else {
      g_timer.reset(new hook_time_info());
      g_timer->flags = timeflags(flags);
    }
  }
}

NVCD_EXPORT void libnvcd_time_report() {
  std::stringstream ss;
  for (const auto& region_entries: g_time_records) {
    for (const auto& dump: region_entries.dumps) {
      ss << dump;
    }
  }
  printf("%s\n", ss.str().c_str());
  g_time_records.clear();
}

NVCD_EXPORT void libnvcd_begin(const char* region_name) {
  // a null region name is totally useless,
  // and will also likely create a segfault,
  // so we may as well enforce non-null input.
  ASSERT(region_name != nullptr);
  ASSERT(strlen(region_name) <= 256);
  if (region_name != nullptr) {
    strncpy(g_region_buffer, region_name, 255);
    g_enabled = true;
    if (g_timer) {
      g_timer->begin_region(region_name);
    }
  }
}

NVCD_EXPORT void libnvcd_end() {
  // g_enabled == false implies a significant flaw
  // in the program logic of the caller.
  // It also opens the door to further errors that
  // coulud arise internally in the future.
  ASSERT(g_enabled == true);
  if (g_enabled) {
    if (g_timer) { 
      g_timer->end_region();
      g_timer->record();   
    }
    // make sure this is set to false before
    // reset_timer() is called
    g_enabled = false;
    reset_timer();
    hook_run_info::num_runs = 0;
  }
}

C_LINKAGE_END
