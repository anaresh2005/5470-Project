// Entry point for code produced by the StragglAR code generator.
//
// Single-process style, same as stragglar/stragglar/allreduce_4GPU.cu:
// one process drives all GPUs, comms come from ncclCommInitAll.
//
// How to call it:
//   - chunkSize = size / (numRanks - 1).
//     Each d_buffers[r] holds (numRanks - 1) chunks; chunk c of rank r
//     starts at d_buffers[r] + c * chunkSize.
//   - d_tempbufs[r] must hold at least chunkSize floats (scratch for "add" ops).
//   - Before calling: run the ReduceScatter among ranks 0..numRanks-2 and
//     launch any straggler delay, as stragglar_allreduce_delay does.
//   - The function only enqueues work on streams[r]. The caller synchronizes
//     the streams and does all timing.
//   - On return (after sync), every d_buffers[r] holds the full AllReduce sum.

#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <nccl.h>

void gen_stragglar_allreduce(float** d_buffers, float** d_tempbufs, int* devs,
                             cudaStream_t* streams, ncclComm_t* comms,
                             int numRanks, size_t chunkSize);
