#pragma once
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
namespace checked {
inline void cuda(cudaError_t status, const char* call, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA failure at line %d (%s): %s\n", line, call, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}
inline void nccl(ncclResult_t status, const char* call, int line) {
  if (status != ncclSuccess) {
    std::fprintf(stderr, "NCCL failure at line %d (%s): %s\n", line, call, ncclGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}
}
#define CUDA(call) checked::cuda((call), #call, __LINE__)
#define NCCL(call) checked::nccl((call), #call, __LINE__)
void handwritten_allreduce(float**, float**, int*, cudaStream_t*, ncclComm_t*, size_t);
