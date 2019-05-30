#define NVCD_HEADER_IMPL
#include <nvcd/nvcd.cuh>
#undef NVCD_HEADER_IMPL

#include <vector>
#include <string>
#include <sstream>

#define DEVICE_BLOCK "\n===================================\n"
#define SECTION_BLOCK "\n----------------------------------\n"

int main(void) {
  nvcd_init();
  
  nvcd_device_info::ptr_type info =
    nvcd_host_get_device_info();

  std::stringstream ss;

  auto print_metrics = [&ss](const std::string& title,
                             const std::vector<nvcd_device_info::metric_entry>& metrics) -> void {
    ss << SECTION_BLOCK
    << title
    << SECTION_BLOCK;
      
    for (size_t j = 0; j < metrics.size(); ++j) {
      ss << "[" << j << "]: " << metrics.at(j).name << "\n";

      for (size_t k = 0; k < metrics.at(j).events.size(); ++k) {
        ss << "\t[" << k << "]: " << metrics.at(j).events.at(k) << "\n";
      }
    }
  };

  auto print_events = [&ss](const std::string& title,
                            const std::vector<std::string>& events) -> void {


    ss << SECTION_BLOCK
    << title
    << SECTION_BLOCK;

    for (size_t j = 0; j < events.size(); ++j) {
      ss << "[" << j << "]: " << events.at(j) << "\n";
    }
    
  };
  
  for (size_t i = 0; i < info->device_names.size(); ++i) {
    nvcd_device_info::name_list_type& events =
      info->events[info->device_names[i]];

    {
      std::vector<std::string> supported;
      std::vector<std::string> unsupported;

      for (const auto& entry: events) {
        if (entry.supported) {
          supported.push_back(entry.name);
        } else {
          unsupported.push_back(entry.name);
        }
      }

      ss << DEVICE_BLOCK
         << "[" << i << "]: " << info->device_names[i];

      print_events("EVENTS - SUPPORTED", supported);
      
      print_events("EVENTS - UNSUPPORTED", unsupported);

      ss << SECTION_BLOCK;
    }

    {
      
      std::vector< nvcd_device_info::metric_entry > supported;
      std::vector< nvcd_device_info::metric_entry > unsupported;

      auto& metrics = info->metrics[info->device_names[i]];
      
      for (size_t k = 0; k < metrics.size(); ++k) {
        if (metrics[k].supported) {
          supported.push_back(metrics[k]);
        } else {
          unsupported.push_back(metrics[k]);
        }
      }
      
      print_metrics("METRICS - SUPPORTED", supported);
      print_metrics("METRICS - UNSUPPORTED", unsupported);
    }
  }

  printf("%s\n", ss.str().c_str());

  nvcd_terminate();

  return 0;
}
