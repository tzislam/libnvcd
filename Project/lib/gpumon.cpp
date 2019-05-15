#include "gpumon.h"

#include <vector>
#include <memory>

static clock64_t* host_ttime = nullptr;

static std::unique_ptr<int> mp(nullptr);

EXTC HOST void gpumon_host_start(int num_threads) {
	mp.reset(new int(5));
  (void)host_ttime;
}

EXTC HOST void gpumon_host_end() {

}
