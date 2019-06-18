#include "nvcd/commondef.h"
#include "nvcd/util.h"

#include <dlfcn.h>

#include <vector_types.h>
#include <vector>
#include <memory>

#define CUDA_HOOK_STUB printf("[HOOK %s:%s]", __FUNC__, __FILE__)

#define CUDA_HOOK(func_name, body, args...)
extern "C" {                                                            \
  typedef cudaError_t (*func_name##_fn_t)(__VA_ARGS__);                 \
  static func_name##_fn_t real_##func_name = NULL;                      \
  NVCD_EXPORT cudaError_t func_name(__VA_ARGS__) {                      \
    if (real_##func_name == NULL) {                                     \
      real_##func_name = (func_name##_fn_t) dlsym(RTLD_NEXT, #func_name); \
    }                                                                   \
    body                                                                \
  }                                                                     \
}

CUDA_HOOK(cudaLaunch,
          {
            CUDA_HOOK_STUB;
            return (*real_cudaLaunch)(entry);
          },
          const void* entry);

struct call_state {
  std::vector<void*> args;
  dim3 grid;
  dim3 block;
  size_t shared_mem;
  cudaStream_t stream;

  call_state(dim3 g, dim3 b, size_t shared_mem_, cudaStream_t stream_)
    : grid(g),
      block(b),
      shared_mem(shared_mem_),
      stream(stream_)
  {}

  ~call_state()
  {}
};

static std::unique_ptr<call_state> g_call_state(nullptr);

CUDA_HOOK(cudaConfigureCall,
          {
            g_call_state.reset(new call_state(grid_dim, block_dim));
            CUDA_HOOK_STUB;
            return (*real_cudaConfigureCall)(grid_dim, block_dim, shared_mem, stream);
          },
          dim3 grid_dim, dim3 block_dim, size_t shared_mem, cudaStream_t stream);


