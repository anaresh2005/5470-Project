// Reuse the upstream n=4 implementation verbatim, including its internal syncs.
// Separate translation unit avoids the generated reduce_add symbol collision.
#define main stragglar_original_main
#include "../stragglar/stragglar/allreduce_4GPU.cu"
#undef main
#include "common.h"

void handwritten_allreduce(float** buffers, float** scratch, int* devs,
                           cudaStream_t* streams, ncclComm_t* comms, size_t chunk) {
  // The upstream helper records an otherwise unused stop event. The harness
  // measures wall time through completion on ALL four streams instead.
  CUDA(cudaSetDevice(devs[0]));
  static cudaEvent_t start = nullptr, stop = nullptr;
  if (!start) {
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&stop));
  }
  stragglar_allreduce_helper(buffers, scratch, devs, streams, comms,
                            start, stop, 4, static_cast<int>(chunk));
}
