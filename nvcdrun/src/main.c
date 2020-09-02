
#include "gpu.h"

int main() {

  if (false) { 
    for (int i = 0; i < 8; ++i) {
      gpu_call(i, 1);
    }
  }
  else {
    gpu_call(7, 100);
  }
  return 0;
}

