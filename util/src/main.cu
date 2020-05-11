#define NVCD_HEADER_IMPL
#include <nvcd/nvcd.cuh>
#undef NVCD_HEADER_IMPL

#include <vector>
#include <string>
#include <sstream>

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

uint32_t parse_uint() {
  long int n = strtol(optarg, nullptr, 10);
  if (n <= 0) {
    exit_with_help(EBAD_INPUT);
  }
  return static_cast<uint32_t>(n);
} 

void parse_args(int argc, char** argv) {
  int opt;
  while ((opt = getopt(argc, argv, "d:n:h")) != -1) {
    switch (opt) {
    case 'h':
      exit_with_help(EHELP);
      break;
    case 'd':
      g_dev = parse_uint();
      if (g_dev > 3) {
	printf("invalid -d option: %" PRIu32 ". device index must be in the range [0, 3]\n",
	       g_dev);
	exit_with_help(EHELP);
      }
      break;
    case 'n':
      g_max = parse_uint();
      break;
    default:
      puts("Unrecognized input");
      exit_with_help(EBAD_INPUT);
      break;
    }
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  
  nvcd_init();
  
  nvcd_device_info::ptr_type info =
    nvcd_host_get_device_info();

  info->multiplex(g_dev, g_max);

  // The following just prints out additional information on metrics and events.
  // Not particulalry important at this time, but it's kept in the event
  // that it might be useful. Should probably be deleted if it isn't used
  // anytime soon though.
  
#if 0
  std::stringstream ss;

  auto print_metrics = [&ss](const std::string& title,
                             const std::vector<nvcd_device_info::metric_entry>& metrics) -> void {
    ss << SECTION_BLOCK
       << title
       << SECTION_BLOCK;
      
    for (size_t j = 0; j < metrics.size(); ++j) {
      ss << "[" << j << "]: " << metrics.at(j).name << "\n";

      for (auto& event_name: metrics.at(j).events) {
        ss << "\tevent IDs corresponding to \"" << event_name.first << "\": ";

        if (event_name.second.size() == 1) {
          ss << STREAM_HEX(4) << event_name.second.at(0) << std::dec << "\n";
        } else {
          ss << "\n";
          for (auto& eid: event_name.second) {
            ss << "\t\t" << STREAM_HEX(4) << eid << std::dec << "\n";
          }
        }
      }
    }
  };

  auto print_events = [&ss](const std::string& title,
                            const std::vector<std::string>& events) -> void {


    ss << SECTION_BLOCK
       << title
       << "|Count = " << events.size() 
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
         << "[" << i << "]: "
         << info->device_names[i];

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
#endif



  nvcd_terminate();

  return 0;
}
