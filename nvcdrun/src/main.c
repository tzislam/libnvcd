
#include "gpu.h"

int main() {

  if (true) { 
    for (int i = 0; i < 8; ++i) {
      gpu_call(i);
    }
  }
  else {
    gpu_call(7);
  }

  return 0;
}

