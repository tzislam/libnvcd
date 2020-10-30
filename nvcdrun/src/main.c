#include "gpu.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <thread>
#include <mutex>
#include <iostream>
#include <unistd.h>
#include <sys/syscall.h>


#define CUDA_DRIVER_FN(expr) cuda_driver_error_print_exit(expr, __LINE__, __FILE__, #expr)
#define CUDA_RUNTIME_FN(expr) cuda_runtime_error_print_exit(expr, __LINE__, __FILE__, #expr)

static void cuda_runtime_error_print_exit(cudaError_t status,
                                          int line,
                                          const char* file,
                                          const char* expr) {
  if (status != cudaSuccess) {
    printf("CUDA RUNTIME: %s:%i:'%s' failed. [Reason] %s:%s\n",
           file,
           line,
           expr,
           cudaGetErrorName(status),
           cudaGetErrorString(status));
      
    exit(1);
  }
}

static void cuda_driver_error_print_exit(CUresult status,
                                         int line,
                                         const char* file,
                                         const char* expr) {
  if (status != CUDA_SUCCESS) {
    printf("CUDA DRIVER: %s:%i:'%s' failed. [Reason] %i\n",
           file,
           line,
           expr,
           status);
      
    exit(1);
  }
}

static inline uint64_t get_thread_id(void) {
  return (uint64_t)syscall(SYS_gettid);
}
enum class test_mode
{
  mt = 0,
  st,
  timed,
  timed_all,
};

static const test_mode test = test_mode::mt;

int main() {
  const unsigned rep = 10;
  const unsigned tflags = 7;
  std::mutex lock;
  CUDA_DRIVER_FN(cuInit(0));
  auto thread_fn = [&lock, &rep, &tflags](int device) {
    CUDA_RUNTIME_FN(cudaSetDevice(device));
    CUdevice cu_device;
    CUDA_DRIVER_FN(cuDeviceGet(&cu_device, device));
    CUcontext cu_context;
    CUDA_DRIVER_FN(cuCtxCreate(&cu_context, 0, cu_device));
    CUDA_DRIVER_FN(cuCtxSetCurrent(cu_context));
    {
      std::lock_guard<std::mutex> guard(lock);
      std::cout << "Sys Thread: " << get_thread_id() << ", Thread: " << std::this_thread::get_id() << ", Device: " << device
                << ", CU Device: " << cu_device << ", CU Context: " << std::hex
                << cu_context << std::dec << std::endl;
    }
    gpu_call(tflags, rep); 
  };
  switch (test) {
  case test_mode::mt: {
    std::cout << "TEST MODE: MULTI-THREADED" << std::endl;
    std::thread t0(thread_fn, 0);
    std::thread t1(thread_fn, 1);
    std::thread t2(thread_fn, 2);
    std::thread t3(thread_fn, 3);
    t0.join();
    t1.join();
    t2.join();
    t3.join();
  } break;
  case test_mode::timed_all:
    std::cout << "TEST MODE: TIMED ALL" << std::endl;
    for (int i = 0; i < 8; ++i) {
      gpu_call(i, 1);
    }
    break;
  case test_mode::st:
    std::cout << "TEST MODE: SINGLE-THREADED" << std::endl;
    thread_fn(0);
    break;
  case test_mode::timed:
    std::cout << "TEST MODE: TIMED" << std::endl;
    gpu_call(7, rep);
    break;
  }
  return 0;
}

