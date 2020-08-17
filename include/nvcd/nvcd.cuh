#ifndef __NVCD_CUH__
#define __NVCD_CUH__

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

#define STREAM_HEX(bytes) "0x" << std::uppercase << std::setfill('0') << std::setw((bytes) << 1) << std::hex

#define DEV_PRINT_PTR(v) msg_verbosef("&(%s) = %p, %s = %p\n", #v, &v, #v, v)

#ifndef NVCD_OMIT_STANDALONE_EVENT_COUNTER

#define NVCD_KERNEL_EXEC_KPARAMS_2(kname, kparam_1, kparam_2, ...)      \
  do {                                                                  \
    cupti_event_data_begin(nvcd_get_events());                          \
    while (!nvcd_host_finished()) {                                     \
      kname<<<kparam_1, kparam_2>>>(__VA_ARGS__);                       \
      CUDA_RUNTIME_FN(cudaDeviceSynchronize());                         \
      g_run_info->run_kernel_count_inc();				\
    }                                                                   \
    cupti_event_data_end(nvcd_get_events());                            \
    NVCD_KERNEL_EXEC_METRICS_KPARAMS_2(nvcd_get_events(),               \
				       kname,				\
				       kparam_1,			\
				       kparam_2,			\
				       __VA_ARGS__);			\
  } while (0)

  #else

#define NVCD_KERNEL_EXEC_KPARAMS_2(kname, kparam_1, kparam_2, ...)      \
  do {                                                                  \
    NVCD_KERNEL_EXEC_METRICS_KPARAMS_2(nvcd_get_events(),               \
				       kname,				\
				       kparam_1,			\
				       kparam_2,			\
				       __VA_ARGS__);			\
  } while (0)

#endif // NVCD_OMIT_STANDALONE_EVENT_COUNTER

#define NVCD_KERNEL_EXEC_METRICS_KPARAMS_2(p_event_data, kname, kparam_1, kparam_2, ...) \
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
        kname<<<kparam_1, kparam_2>>>(__VA_ARGS__);                     \
        CUDA_RUNTIME_FN(cudaDeviceSynchronize());                       \
        g_run_info->run_kernel_count_inc();				\
      }                                                                 \
                                                                        \
      cupti_event_data_end(&__e->metric_data->event_data[i]);           \
    }                                                                   \
  } while (0)                                                           

namespace detail {
  DEV clock64_t* dev_tstart = nullptr;
  DEV clock64_t* dev_ttime = nullptr;
  DEV int* dev_num_iter = nullptr;
  DEV uint* dev_smids = nullptr;
}

// Hook management

extern "C" {  
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

  uint32_t exec_count;
  
  kernel_invoke_data(size_t num_threads_)
    : times(num_threads_, 0),
      smids(num_threads_, 0),
      time_stddev(0.0),
      num_threads(num_threads_),
      exec_count(0)
  {}

  ~kernel_invoke_data()
  {}

  void fill_outliers(double bounds,
                     double q1,
                     double q3,
		     const std::string& threshold,
                     const std::vector<block>& in,
                     std::vector<block>& out) {
    
    clock64_t max = static_cast<clock64_t>(q3) + static_cast<clock64_t>(bounds);

    #if 0
    msg_userf(STATISTICS_TAG " [ticks measured for the region]{q1, q3, bounds, %s outlier threshold} = {%f, %f, %f, %" PRId64 "}\n",
    	      threshold.c_str(),
    	      q1, q3, bounds, max);
    #endif

    for (const block& b: in) {
      if (b.time > max) {
        out.push_back(b);
      }
    }
  }

  void print_blockv(const char* name, const std::vector<block>& v) {
    msg_verbosef("=====%s=====\n", name);
    
    for (size_t i = 0; i < v.size(); ++i) {
      msg_verbosef("[%lu] time = %" PRId64 " , thread = %" PRId32 "\n",
		   i,
		   v[i].time,
		   v[i].thread);
    }
  }
  
  void write() {

    std::unordered_set<int> smids_used;
    
    for (size_t i = 0; i < num_threads; ++i) {
      smids_used.insert(smids[i]);
    }

    msg_verbosef("Number of streaming multiprocessors used: %" PRIu64 "\n", smids_used.size());
    
    {
      std::vector<block> sorted;

      for (size_t i = 0; i < num_threads; ++i) {
        sorted.push_back(block{static_cast<int>(i), times[i]}); 
      }
      
      std::sort(sorted.begin(), sorted.end(), [](const block& a, const block& b) -> bool {
						return a.time < b.time;
					      });

      size_t qlen = num_threads >> 2;

      double q1 = static_cast<double>(sorted[qlen].time)
        + static_cast<double>(sorted[qlen - 1].time);
      q1 = q1 * 0.5;

      double q3 = static_cast<double>(sorted[qlen * 3].time)
        + static_cast<double>(sorted[(qlen * 3) - 1].time);
      q3 = q3 * 0.5;

      double iqr = q3 - q1;

      double minorb = iqr * 1.5;
      fill_outliers(minorb, q1, q3, "minor", sorted, load_minor);

      double majorb = iqr * 3.0;
      fill_outliers(majorb, q1, q3, "major", sorted, load_major);
    }

    msg_verbosef("Major outlier thread block count: %" PRId64 "\n", load_major.size());
    msg_verbosef("Minor outlier thread block count: %" PRId64 "\n", load_minor.size());
    
    //print_blockv("load_minor", load_minor);
    //print_blockv("load_major", load_major);
  }
};

template <typename T>
NVCD_CUDA_EXPORT void __buf_to_vec(std::vector<T>& vec, T* buf, uint32_t length) {
  vec.resize(length);
  memcpy(&vec[0], &buf[0], sizeof(buf[0]) * length);
  free(buf);
}

template <class T>
struct cupti_unset {
  static_assert(std::is_integral<T>::value,
		"CUpti handle should be numeric to some degree. Otherwise, an alternative will be necessary");

  static const T value;
};

template <class T>
const T cupti_unset<T>::value = std::numeric_limits<T>::max(); 

using event_list_type = std::vector<CUpti_EventID>;

struct event_group {
  event_list_type events;
  CUpti_EventGroup group;
};

using event_group_list_type = std::vector<event_group>;

//
// these two routines are used for testing and nothing more.
//
NVCD_CUDA_EXPORT bool cmp_events(const event_list_type& a, const event_list_type& b) {
  ASSERT(a.size() == b.size());
  bool bfound = true;
  size_t i = 0; 
  while (bfound && i < a.size()) {
    bool found = false;
    size_t j = 0;
    while (!found && j < b.size()) {
      found = a.at(i) == b.at(j);
      j++;
    }
    bfound = found;
    i++;
  }
  return bfound;
}

NVCD_CUDA_EXPORT bool operator == (const event_group& a, const event_group& b) {
  return
    (a.events.size() == b.events.size())
    ? cmp_events(a.events, b.events)
    : false;
}

NVCD_CUDA_EXPORT void print_path_error(const std::string& fn,
		      const std::string& path) {
  std::string err(fn);
  err.append(" on \'");
  err.append(path);
  err.append("\'");	     
  perror(err.c_str());
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
      : entry(name_, id_unset, supported_)
        
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
  
  template <class S,
	    class T>
  using cupti_enum_fn_type = CUptiResult (*)(S, size_t*, T*);

  template <class S>
  using cupti_get_num_fn_type = CUptiResult (*)(S, uint32_t*);
  
  template <class S,
	    class T,
	    bool assertFull>
  struct cupti_data_enum {
    using src_type = S;
    using dst_buf_type = T;
    using enum_fn_type = cupti_enum_fn_type<src_type, dst_buf_type>;
    using get_num_fn_type = cupti_get_num_fn_type<src_type>;
    
    template <get_num_fn_type get_fn,
	      enum_fn_type enum_fn>
    static void fill(src_type src,
		     std::vector<dst_buf_type>& v) {
      uint32_t cnt = 0;
      CUPTI_FN(get_fn(src, &cnt));

      v.resize(cnt, cupti_unset<dst_buf_type>::value);

      size_t sz = sizeof(dst_buf_type) * static_cast<size_t>(cnt);
      size_t sz_o = sz;
      
      CUPTI_FN(enum_fn(src, &sz, v.data()));

      if (assertFull) {
	ASSERT(sz == sz_o);
      }    
    } 
  };

  using cupti_device_domain_enum_t =
    cupti_data_enum<CUdevice,
		    CUpti_EventDomainID,
		    true>;
  
  using cupti_domain_event_enum_t =
    cupti_data_enum<CUpti_EventDomainID,
		    CUpti_EventID,
		    true>;

  using cupti_attr_str_t = std::array<char, 128>;
    
  class domain_group_gen {
    event_list_type events;
    event_group_list_type groupings;
    const size_t max_events_per_group;
    const CUdevice device;
    const CUcontext context;
    const CUpti_EventDomainID domain;    

    void load_events() {      
      cupti_domain_event_enum_t::fill<&cuptiEventDomainGetNumEvents,
				      &cuptiEventDomainEnumEvents>(domain,
								   events);      
      msg_userf(INFO_TAG "\tNumber of events available in this domain: %" PRIu64 "\n", events.size());      
    }   

    bool try_events(event_list_type e, std::vector<CUpti_EventGroup>& ptrs) {      
      ASSERT(!e.empty());
      
      CUpti_EventGroup group = nullptr;
      CUPTI_FN(cuptiEventGroupCreate(context, &group, 0));
      CUptiResult result = CUPTI_SUCCESS;
      
      size_t i = 0;
      while (i < e.size() && result == CUPTI_SUCCESS) {
	result = cuptiEventGroupAddEvent(group, e.at(i));
	if (result == CUPTI_SUCCESS) {
	  i++;
	}
      }
      ASSERT((
	      (i == e.size())
	      &&
	      (result == CUPTI_SUCCESS)
	      )
	     ||
	     (
	      (i < e.size())
	      &&
	      (result != CUPTI_SUCCESS)
	      ));
      
      if (result == CUPTI_SUCCESS) {
	ptrs.push_back(group);
      } else {	
	CUPTI_FN(cuptiEventGroupDestroy(group));
      }

      return result == CUPTI_SUCCESS;
    }

    void find_groups() {
      // greedy brute force algorithm to grab as many events as possible
      // for a single group. events are not duplicated across group
      // combinations.
      bool grouped = false;
      std::vector<int> visited;
      auto in_visited =
	[&visited](int i) -> bool {	  
	  for (int k : visited) if (i == k) return true;
	  return false;
	};
      while (!grouped) {
	event_list_type try_group{};
	std::vector<CUpti_EventGroup> ptrs;
	uint32_t i = 0;
	while (i < events.size() && try_group.size() < max_events_per_group) {
	  if (!in_visited(i)) {
	    try_group.push_back(events.at(i));
	    if (try_events(try_group, ptrs)) {
	      visited.push_back(i);
	    } else {
	      try_group.pop_back();
	    }	    	    
	  }
	  i++;
	}
	if (!ptrs.empty()) {
	  CUpti_EventGroup largest = ptrs.back();
	  // 'largest' is a group that contains the events
	  // that these other groups contain, so we can do
	  // without them.
	  ptrs.pop_back();
	  for (CUpti_EventGroup less: ptrs) {
	    CUPTI_FN(cuptiEventGroupDestroy(less));
	  }	  
	  groupings.push_back({ try_group, largest });
	}
	grouped = visited.size() == events.size();
      }

      // just a good faith brute force test to
      // ensure we haven't duplicated events across multiple groups.
      constexpr bool do_the_test = true;
      if (do_the_test) {
	// ensure we have the same size in the CSV file
	volatile size_t count = 0;
	for (const auto& g: groupings) {
	  count += g.events.size();
	}
	ASSERT(count == events.size());

	// ensure no duplicates; assuming both of these are true,
	// we have what we want.
	for (volatile size_t i = 0; i < groupings.size(); i++) {
	  for (volatile size_t j = 0; j < groupings.size(); ++j) {
	    if (i != j) {
	      for (const auto& e0: groupings.at(i).events) {
		for (const auto& e1: groupings.at(j).events) {
		  ASSERT(e0 != e1);
		}
	      }
	    }
	  }
	}
      }
    }
    
  public:
    domain_group_gen(CUdevice device, CUcontext context, CUpti_EventDomainID domain, size_t max_events)
      : max_events_per_group(max_events),
	device(device),
	context(context),
	domain(domain) {

      load_events();
      find_groups();
    }

    event_group_list_type operator()() {
      return groupings;
    }
  };

  cupti_attr_str_t event_name(CUpti_EventID e) {
    cupti_attr_str_t r{};
    r.fill(0);
    size_t sz = r.size() * sizeof(r[0]);
    CUPTI_FN(cuptiEventGetAttribute(e, CUPTI_EVENT_ATTR_NAME, &sz, r.data()));
    return r;
  }

  cupti_attr_str_t event_domain_name(CUpti_EventDomainID e) {
    cupti_attr_str_t r{};
    r.fill(0);
    size_t sz = r.size() * sizeof(r[0]);
    CUPTI_FN(cuptiEventDomainGetAttribute(e, CUPTI_EVENT_DOMAIN_ATTR_NAME, &sz, r.data()));
    ASSERT(sz < r.size());
    return r;
  }
  
  void cupti_domain_csv_write(const cupti_attr_str_t& domain_name,
			      const event_group_list_type& groupings) {
    std::string output_path("./");
    
    output_path.append(domain_name.data());
    output_path.append(".csv");
    
    FILE* f = fopen(output_path.c_str(),
		    "wb");
    
    if (f != nullptr) {
      std::stringstream ss;      
      for (const auto& g: groupings) {
	size_t j = 0;
	for (const auto& e: g.events) {
	  auto name = event_name(e);
	  ss << name.data();
	  if (j < g.events.size() - 1) {
	    ss << ",";
	  }
	  j++;
	}
	ss << "\n";
      }
      fprintf(f, "%s", ss.str().c_str());
    } else {
      
      print_path_error("cupti_domain_csv_write->fopen",
		       output_path.c_str());
    }

  }
  
  void multiplex(uint32_t nvcd_index, uint32_t max_num) {        
    std::vector<CUpti_EventDomainID> domain_buffer{};

    // fill domain buffer with all domain IDs corresponding to
    // the device referenced by nvcd_index.
    cupti_device_domain_enum_t::fill<&cuptiDeviceGetNumEventDomains,
				     &cuptiDeviceEnumEventDomains>(g_nvcd.devices[nvcd_index],
								   domain_buffer);


    puts("=======multiplex=========");

    size_t smax_num =
      max_num != UINT32_MAX
      ? static_cast<size_t>(max_num)
      : std::numeric_limits<size_t>::max();
    
    for (CUpti_EventDomainID domain: domain_buffer) {
      cupti_attr_str_t domain_name = event_domain_name(domain);
      
            
      msg_userf(INFO_TAG "Processing domain: %s\n", domain_name.data());
      
      domain_group_gen generator(g_nvcd.devices[nvcd_index],
				 g_nvcd.contexts[nvcd_index],
				 domain,
				 smax_num);

      auto groupings = generator();

      cupti_domain_csv_write(domain_name,
			     groupings);

      // user output
      constexpr bool verbose = false;
      if (verbose) {
	std::stringstream ss;
	ss << "---\n\tnum combinations: " << groupings.size() << "\n---\n";
	size_t i = 0;
	for (const auto& group: groupings) {
	  ss << "\tgroup[" << (i + 1) <<"] = { ";
	  size_t j = 0;
	  for (const auto& e: group.events) {
	    cupti_attr_str_t buffer = event_name(e);	      
	    
	    ss << static_cast<const char*>(&buffer[0])
	       << "("
	       << STREAM_HEX(sizeof(e))
	       << e
	       << std::dec
	       << ")";

	    if (j < (group.events.size() - 1)) {
	      ss << ", ";
	    }
	    j++;
	  }
	  ss << " }\n";
	  i++;
	}
	msg_verbosef("%s\n", ss.str().c_str());
      }      
    }    
  } 
  
  nvcd_device_info() {
    ASSERT(g_nvcd.initialized == true);
    
    for (auto i = 0; i < g_nvcd.num_devices; ++i) {
      std::string device(g_nvcd.device_names[i]);
      CUdevice device_handle = g_nvcd.devices[i];

      // TODO:
      // this really should be created outside of the standard global framework,
      // using a "one-shot" kind of allocation that can be freely allocated
      // using strictly cupti_event_data functions.
      // This way we don't need to call nvcd_get_events()
      // and worry about how the underlying state is affected.
      nvcd_init_events(g_nvcd.devices[i], g_nvcd.contexts[i]);
            
      device_names.push_back(device);

      cupti_event_data_t* global = nvcd_get_events();

      size_t num_event_names = 0;
      char** event_names = cupti_get_event_names(global, &num_event_names);
      
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

      free_strlist(event_names, num_event_names);

      // will be set to NULL, so the next call to nvcd_init_events() will be OK.
      // note that every time nvcd_init_events() is called, the cupti_event_data_t*
      // instance that's allocated needs to be "nulled" out before nvcd_init_events() is called again.
      // we do this with cupti_event_data_set_null()
      cupti_event_data_free(global);
    }
  }
};


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

extern struct nvcd_run_info* g_run_info;

struct nvcd_run_info {
  std::vector<kernel_invoke_data> kernel_stats;
  
  counter_map_type counters_start;
  counter_map_type counters_end;
  counter_map_type counters_diff;

  std::string region_name;
  
  size_t curr_num_threads;
  const char* func_name;
  uint32_t run_kernel_exec_count;

  static size_t num_runs;
  
  nvcd_run_info()
    : curr_num_threads(0),
      func_name(nullptr),
      run_kernel_exec_count(0) {
  }

  ~nvcd_run_info() {
  }

  void run_kernel_count_inc() {
    run_kernel_exec_count++;
  }
  
  void update() {
    ASSERT(curr_num_threads != 0);
    
    if (false) {        
      kernel_invoke_data d(curr_num_threads);

      nvcd_device_get_smids(&d.smids[0]);
      nvcd_device_get_ttime(&d.times[0]);
      
      d.exec_count = run_kernel_exec_count;
      
      kernel_stats.push_back(std::move(d));
    }

    curr_num_threads = 0;
    run_kernel_exec_count = 0;
   
    cupti_event_data_t* global = nvcd_get_events();
    // we do this to compute the difference
    // from the previous run
    for (const auto& kv: counters_end) {
      counters_start[kv.first] = kv.second;      
    }
    
    cupti_event_data_enum_event_counters(global, nvcd_run_info::enum_event_counters);    
    
    counters_diff = counters_end - counters_start;

    num_runs++;
  }

  static bool enum_event_counters(cupti_enum_event_counter_iteration_t* it) {    
    if (g_run_info->counters_end[it->event].empty()) {
      g_run_info->counters_end[it->event].resize(it->num_instances, 0);
    }
    ASSERT(it->instance < it->num_instances);
    g_run_info->counters_end[it->event][it->instance] += it->value;
    return true;
  }
  
  void report() {
    ASSERT(num_runs > 0);
    
    msg_userf("================================ invocation %" PRIu64 " for \'%s\' ================================\n",
	      num_runs - 1,
	      region_name.c_str());
   
    if (!kernel_stats.empty()) kernel_stats.at(num_runs - 1).write();

    std::stringstream ss;
    msg_verbosef("counters_diff size: %" PRIu64 "\n", counters_diff.size());
    for (const auto& kv : counters_diff) {
      const auto& key = kv.first;
      const auto& value = kv.second;
      char* event_name = cupti_event_get_name(key);
      ASSERT(event_name != nullptr);
      double avg = 0;
      uint64_t summation = 0;
      uint64_t maximum = 0; // the lowest possible count
      uint64_t minimum = 10000000; //something very large so that it changes
      uint64_t temp_var;
      for (size_t index = 0; index < value.size(); ++index) {
	temp_var = value.at(index);
	summation += temp_var;
	avg += temp_var;
	maximum = (maximum < temp_var) ? temp_var : maximum;
	minimum = (minimum > temp_var) ? temp_var : minimum;
	//ss << "|COUNTER|" << region_name << ":" << event_name << ":" << value.at(index) << "\n";
      }
      avg /= value.size();
      //ss << "|COUNTER|" << region_name << ":" << event_name << ":" << avg << "\n";
      ss << "|COUNTER|" << region_name << ":" << event_name << ": SUM: " << summation << " AVG: " << avg << " MAX: " << maximum << " MIN: " << minimum << "\n";
      free(event_name);
    }
    
    msg_userf("%s", ss.str().c_str());
    
    cupti_report_event_data(nvcd_get_events());
  }
};

size_t nvcd_run_info::num_runs = 0;

extern nvcd_run_info* g_run_info;

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

    msg_verbosef("HOST-SIDE DEVICE ADDRESS FOR %s: %p. value: %p\n",
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
    ASSERT(g_run_info != nullptr);
    
    g_run_info->report();
  }

  NVCD_CUDA_EXPORT uint32_t nvcd_last_kernel_exec_count() {
    ASSERT(!g_run_info->kernel_stats.empty());
    return g_run_info->kernel_stats[g_run_info->kernel_stats.size() - 1].exec_count;
  }
  
  NVCD_CUDA_EXPORT void nvcd_init() {
    nvcd_init_cuda();

    if (g_run_info == nullptr) {
      g_run_info = new nvcd_run_info();
    }
        
    ASSERT(g_nvcd.initialized == true);
    ASSERT(g_run_info != nullptr);
  }

  NVCD_CUDA_EXPORT void nvcd_host_begin(const char* region_name, int num_cuda_threads) {     
    nvcd_init();

    g_run_info->region_name = std::string(region_name);

    ASSERT(g_nvcd.initialized == true);
    ASSERT(g_run_info != nullptr);

    nvcd_device_init_mem(num_cuda_threads);

    g_run_info->curr_num_threads = static_cast<size_t>(num_cuda_threads);

    nvcd_init_events(g_nvcd.devices[0],
                     g_nvcd.contexts[0]);
  }

  NVCD_CUDA_EXPORT bool nvcd_host_finished() {
    return cupti_event_data_callback_finished(nvcd_get_events());
  }

  NVCD_CUDA_EXPORT void nvcd_terminate();

  NVCD_CUDA_EXPORT void nvcd_host_end() {
    ASSERT(g_nvcd.initialized == true);
    
    nvcd_calc_metrics();

    g_run_info->update();

    nvcd_report();
    
    nvcd_device_free_mem();   

    nvcd_terminate();
  }
  
  NVCD_CUDA_EXPORT nvcd_device_info::ptr_type nvcd_host_get_device_info() {
    ASSERT(g_nvcd.initialized == true);
    nvcd_device_info::ptr_type ptr(new nvcd_device_info());
    return std::move(ptr);
  }

  NVCD_CUDA_EXPORT void nvcd_terminate() {
    nvcd_reset_event_data();
 
    for (int i = 0; i < g_nvcd.num_devices; ++i) {
      ASSERT(g_nvcd.contexts[i] != NULL);
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
}

//
// Define this in only one source file,
// and then include this header right after,
// and then undefine it
//
#ifdef NVCD_HEADER_IMPL

extern "C" {

  size_t dev_tbuf_size = 0;
  size_t dev_num_iter_size = 0;
  size_t dev_smids_size = 0;

  void* d_dev_tstart = nullptr;
  void* d_dev_ttime = nullptr;
  void* d_dev_num_iter = nullptr;
  void* d_dev_smids = nullptr;

  volatile bool test_imbalance_detect = true;
}

nvcd_run_info* g_run_info = nullptr;

template <class SThreadType, 
	  class TKernFunType, 
	  class ...TArgs>
static inline void nvcd_run_metrics(const TKernFunType& kernel, 
				    const SThreadType& block_size, 
				    const SThreadType& threads_per_block,
				    TArgs... args) {
  cupti_event_data_t* __e = nvcd_get_events();                           
  
  ASSERT(__e->is_root == true);                                       
  ASSERT(__e->initialized == true);                                   
  ASSERT(__e->metric_data != NULL);                                   
  ASSERT(__e->metric_data->initialized == true);                      
                                                                        
  for (uint32_t i = 0; i < __e->metric_data->num_metrics; ++i) {      
    cupti_event_data_begin(&__e->metric_data->event_data[i]);         

    while (!cupti_event_data_callback_finished(&__e->metric_data->event_data[i])) {
      kernel<<<block_size, threads_per_block>>>(args...);                       
      CUDA_RUNTIME_FN(cudaDeviceSynchronize());                       
      g_run_info->run_kernel_count_inc();				
    }                                                                 
                                                                        
    cupti_event_data_end(&__e->metric_data->event_data[i]);           
  }                                                                     
}


template <class SThreadType, 
	  class TKernFunType, 
	  class ...TArgs>
static inline void nvcd_run(const TKernFunType& kernel, 
			    const SThreadType& block_size, 
			    const SThreadType& threads_per_block,
			    TArgs... args) {

  
  //  g_run_info->func_name = *(const char**)((uintptr_t)(&kernel) + 8);
  
  cupti_event_data_begin(nvcd_get_events());                          
  while (!nvcd_host_finished()) {                                     
    kernel<<<block_size, threads_per_block>>>(args...);                       
    CUDA_RUNTIME_FN(cudaDeviceSynchronize());                         
    g_run_info->run_kernel_count_inc();				
  }                                                                   
  cupti_event_data_end(nvcd_get_events());    

  nvcd_run_metrics(kernel, block_size, threads_per_block, args...);
}

#endif // NVCD_HEADER_IMPL

#endif // __NVCD_CUH__

