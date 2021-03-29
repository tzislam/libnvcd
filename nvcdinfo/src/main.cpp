#include <inttypes.h>

#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <memory>
#include <iomanip>

#include <nvcd/nvcd.h>
#include <nvcd/util.h>
#include <nvcd/cupti_util.h>

#define STREAM_HEX(bytes) "0x" << std::uppercase << std::setfill('0') << std::setw((bytes) << 1) << std::hex

template <class T>
struct cupti_unset {
  static_assert(std::is_integral<T>::value,
		"CUpti handle should be numeric to some degree. Otherwise, an alternative will be necessary");

  static const T value;
};

template <class T>
const T cupti_unset<T>::value = std::numeric_limits<T>::max(); 

static inline void print_path_error(const std::string& fn,
  const std::string& path) {
  std::string err(fn);
  err.append(" on \'");
  err.append(path);
  err.append("\'");	     
  perror(err.c_str());
}

using event_list_type = std::vector<CUpti_EventID>;

struct event_group {
  event_list_type events;
  CUpti_EventGroup group;
};

using event_group_list_type = std::vector<event_group>;

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
    
    template <get_num_fn_type get_fn, enum_fn_type enum_fn>
    static void fill(src_type src, std::vector<dst_buf_type>& v) {
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
      cupti_domain_event_enum_t::fill<&cuptiEventDomainGetNumEvents, &cuptiEventDomainEnumEvents>(domain, events);      
      msg_userf(INFO_TAG "\tNumber of events available in this domain: %" PRIu64 "\n", 
                events.size());
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
        (result == CUPTI_SUCCESS))
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
      auto in_visited = [&visited](int i) -> bool {	  
	      for (int k : visited) { 
          if (i == k) return true;
        }
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
    
    FILE* f = fopen(output_path.c_str(), "wb");
    
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

#define DEVICE_BLOCK "\n===================================\n"
#define SECTION_BLOCK "\n----------------------------------\n"

static void exit_with_help(int code) {
  puts("Usage:\n"
       "nvcdinfo [-h] [-n $num] [-d $device]\n"
       "\t-d\tUses the device index represented by $device to query event information. If unspecified, 0 will be used. Allowed range is [0, 3].\n"
       "\t-n\tWill only print event groups with sizes that are less than or equal to the integer specified by $num.\n"
       "\t\tNote that if $num is less than or equal to 0, then the program will exit.\n"
       "\t-h\tPrints this help message and exits.");
  exit(code);
}

static uint32_t g_max = UINT32_MAX;
static uint32_t g_dev = 0;

static uint32_t parse_uint(long int min, long int max) {
  long int n = strtol(optarg, nullptr, 10);
  if (!(min <= n && n <= max)) {
    printf("Invalid range specified for integer value %" PRId64 ". Accepted range is (%" PRId64", %" PRId64 ")\n",
	   n,
	   min,
	   max);
    exit_with_help(EBAD_INPUT);
  }
  return static_cast<uint32_t>(n);
} 

static void parse_args(int argc, char** argv) {
  int opt;
  while ((opt = getopt(argc, argv, "d:n:h")) != -1) {
    switch (opt) {
    case 'h':
      exit_with_help(EHELP);
      break;
    case 'd':
      g_dev = parse_uint(0, 3);
      break;
    case 'n':
      g_max = parse_uint(1, 100);
      break;
    default:
      puts("Unrecognized input");
      exit_with_help(EBAD_INPUT);
      break;
    }
  }
}

static void terminate() {
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

int main(int argc, char** argv) {
  parse_args(argc, argv);
  
  nvcd_init_cuda();
  ASSERT(g_nvcd.initialized == true);
  
  nvcd_device_info::ptr_type info(new nvcd_device_info());

  info->multiplex(g_dev, g_max);

  terminate();

  return 0;
}
