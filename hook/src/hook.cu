#include <cuda_runtime_api.h>

#define NVCD_HEADER_IMPL
#include <nvcd/nvcd.cuh>
#undef NVCD_HEADER_IMPL

#include <dlfcn.h>

#include <time.h>

#include <assert.h>

#include <string.h>

#include <cstdint>

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
  ASSERT(__e->metric_data != NULL);                                   
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
    while (result == cudaSuccess && !nvcd_host_finished()) {
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
    if (g_timer) g_timer->begin_kernel();
    nvcd_host_begin(g_region_buffer, gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z);
    ret = nvcd_run2(real_cudaLaunchKernel, func, gridDim, blockDim, args, sharedMem, stream);
    nvcd_host_end();
    if (g_timer) g_timer->end_kernel();
  }
  else {
    printf("[HOOK OFF %s]\n", __FUNC__);
    ret = real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
  }
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
    nvcd_run_info::num_runs = 0;
  }
}

C_LINKAGE_END
