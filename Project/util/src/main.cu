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
         << "[" << i << "]: " << info->device_names[i]
         << SECTION_BLOCK
         << "EVENTS - SUPPORTED"
         << SECTION_BLOCK;

      for (size_t j = 0; j < supported.size(); ++j) {
        ss << "[" << j << "]: " << supported[j] << "\n";
      }

      ss << SECTION_BLOCK
         << "EVENTS - UNSUPPORTED"
         << SECTION_BLOCK;
    
      for (size_t j = 0; j < unsupported.size(); ++j) {
        ss << "[" << j << "]: " << unsupported[j] << "\n";
      }

      ss << SECTION_BLOCK;
    }

    {
      ss << SECTION_BLOCK
         << "METRICS"
         << SECTION_BLOCK;

      auto& metrics = info->metrics[info->device_names[i]];
      
      for (size_t j = 0; j < metrics.size(); ++j) {
        ss << "[" << j << "]: " << metrics[j].name << "\n";

        for (size_t k = 0; k < metrics[j].events.size(); ++k) {
          ss << "\t[" << k << "]: " << metrics[j].events[k] << "\n";
        }
      }
    }
  }

  printf("%s\n", ss.str().c_str());

  nvcd_terminate();

  return 0;
}
