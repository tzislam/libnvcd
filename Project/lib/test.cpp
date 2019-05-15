#include "gpumon.h"

int main() {
	gpumon_host_start(1024);

	gpumon_host_end();

	return 0;
}
