#include "common.h"
#include "validation.h"
#include "../gen/gen_stragglar.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Adapted from upstream fill_pattern and gpu_sleep_kernel. clock64 avoids the
// short wraparound of the upstream clock() counter. Delay remains nominal:
// GPU frequency changes can affect the actual elapsed time.
__global__ void fill_pattern(float* dst, size_t count, int rank, uint32_t seed) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) dst[i] = input_value(rank, i, seed);
}
__global__ void gpu_sleep_kernel(unsigned long long cycles) {
  const auto start = clock64();
  while (clock64() - start < cycles) {}
}

static unsigned long long integer(const char* s, const char* name) {
  std::string str(s); size_t end = 0;
  if (str.empty() || str.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error(std::string("invalid ") + name);
  auto result = std::stoull(str, &end);
  if (end != str.size()) throw std::runtime_error(std::string("invalid ") + name);
  return result;
}

int main(int argc, char** argv) try {
  if (argc < 5 || argc > 8) {
    std::fprintf(stderr, "Usage: %s bytes all|nccl|handwritten|generated iters delay_ms [warmups=5] [physical_straggler=3] [seed=5470]\n", argv[0]);
    return 2;
  }
  auto raw_bytes = integer(argv[1], "bytes");
  auto raw_iters = integer(argv[3], "iterations");
  auto raw_warmups = argc > 5 ? integer(argv[5], "warmups") : 5;
  auto raw_straggler = argc > 6 ? integer(argv[6], "physical_straggler") : 3;
  auto raw_seed = argc > 7 ? integer(argv[7], "seed") : 5470;
  size_t end = 0; double delay = std::stod(argv[4], &end);
  if (end != std::string(argv[4]).size() || !std::isfinite(delay) || delay < -1 || delay > 1000)
    throw std::runtime_error("delay_ms must be -1 or between 0 and 1000");
  if (delay < 0 && delay != -1) throw std::runtime_error("only -1 is a valid negative delay");
  if (delay == -1) delay = 0; // launcher compatibility; NEVER skip ReduceScatter.
  if (!raw_bytes || raw_bytes % 12 || raw_bytes > size_t(INT_MAX) * 4 ||
      raw_iters < 1 || raw_iters > 100000 || raw_warmups > 10000 ||
      raw_straggler > 3 || raw_seed > UINT32_MAX)
    throw std::runtime_error("bytes must be positive, divisible by 12, and <= 4*INT_MAX; iters 1..100000; warmups 0..10000; GPU 0..3; seed uint32");
  const size_t bytes = raw_bytes, count = bytes / sizeof(float), chunk = count / 3;
  const int iterations = raw_iters, warmups = raw_warmups, straggler = raw_straggler;
  const uint32_t seed = raw_seed;
  std::string selected = argv[2];
  if (selected == "stragglar") selected = "handwritten";
  if (selected != "all" && selected != "nccl" && selected != "handwritten" && selected != "generated")
    throw std::runtime_error("unknown algorithm");
  int gpu_count = 0; CUDA(cudaGetDeviceCount(&gpu_count));
  if (gpu_count < 4) throw std::runtime_error("four visible CUDA GPUs are required");
  int devs[4] = {0, 1, 2, 3};
  std::swap(devs[straggler], devs[3]); // physical slow GPU becomes logical rank 3.
  cudaStream_t streams[4]; ncclComm_t comms[4], subcomms[3];
  float *buffers[4], *scratch[4];
  NCCL(ncclCommInitAll(comms, 4, devs));
  NCCL(ncclCommInitAll(subcomms, 3, devs)); // excludes devs[3], even after remap.
  for (int r = 0; r < 4; ++r) {
    CUDA(cudaSetDevice(devs[r]));
    CUDA(cudaStreamCreate(&streams[r]));
    CUDA(cudaMalloc(&buffers[r], bytes));
    CUDA(cudaMalloc(&scratch[r], chunk * sizeof(float)));
  }
  int khz = 0, nccl_version = 0, runtime_version = 0, driver_version = 0;
  CUDA(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, devs[3]));
  NCCL(ncclGetVersion(&nccl_version));
  CUDA(cudaRuntimeGetVersion(&runtime_version)); CUDA(cudaDriverGetVersion(&driver_version));
  const auto cycles = static_cast<unsigned long long>(delay * khz);
  std::printf("META,nccl=%d,cuda_runtime=%d,cuda_driver=%d,devices=%d:%d:%d:%d,clock_khz=%d,seed=%u,warmups=%d\n",
              nccl_version, runtime_version, driver_version, devs[0], devs[1], devs[2], devs[3], khz, seed, warmups);
  auto sync = [&]() {
    for (int r = 0; r < 4; ++r) {
      CUDA(cudaSetDevice(devs[r])); CUDA(cudaGetLastError());
      CUDA(cudaStreamSynchronize(streams[r]));
      ncclResult_t error; NCCL(ncclCommGetAsyncError(comms[r], &error)); NCCL(error);
      if (r < 3) { NCCL(ncclCommGetAsyncError(subcomms[r], &error)); NCCL(error); }
    }
  };
  auto reset = [&]() {
    for (int r = 0; r < 4; ++r) {
      CUDA(cudaSetDevice(devs[r]));
      fill_pattern<<<std::min<size_t>((count + 255) / 256, 4096), 256, 0, streams[r]>>>(buffers[r], count, devs[r], seed);
      CUDA(cudaGetLastError());
      CUDA(cudaMemsetAsync(scratch[r], 0, chunk * sizeof(float), streams[r]));
    }
    sync();
  };
  auto run = [&](const std::string& algorithm, bool inject) {
    if (inject && cycles) {
      CUDA(cudaSetDevice(devs[3]));
      gpu_sleep_kernel<<<1, 1, 0, streams[3]>>>(cycles); CUDA(cudaGetLastError());
    }
    if (algorithm == "nccl") {
      NCCL(ncclGroupStart());
      for (int r = 0; r < 4; ++r) {
        CUDA(cudaSetDevice(devs[r]));
        NCCL(ncclAllReduce(buffers[r], buffers[r], count, ncclFloat, ncclSum, comms[r], streams[r]));
      }
      NCCL(ncclGroupEnd());
    } else {
      NCCL(ncclGroupStart());
      for (int r = 0; r < 3; ++r) {
        CUDA(cudaSetDevice(devs[r]));
        NCCL(ncclReduceScatter(buffers[r], buffers[r] + r * chunk, chunk,
                              ncclFloat, ncclSum, subcomms[r], streams[r]));
      }
      NCCL(ncclGroupEnd());
      // Complete the subcommunicator before switching communicators. Only
      // nonstraggler streams are synchronized, so this overlaps the delay.
      for (int r = 0; r < 3; ++r) {
        CUDA(cudaSetDevice(devs[r])); CUDA(cudaStreamSynchronize(streams[r]));
      }
      if (algorithm == "generated")
        gen_stragglar_allreduce(buffers, scratch, devs, streams, comms, 4, chunk);
      else handwritten_allreduce(buffers, scratch, devs, streams, comms, chunk);
    }
    sync();
  };
  std::vector<float> reference(count), actual(count);
  reset(); run("nccl", false);
  for (int r = 0; r < 4; ++r) {
    CUDA(cudaSetDevice(devs[r])); CUDA(cudaMemcpy(actual.data(), buffers[r], bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i)
      if (!close_value(actual[i], expected_value(i, seed)))
        throw std::runtime_error("NCCL reference does not match independent CPU sum");
    if (r == 0) reference = actual;
  }
  std::printf("CHECK,reference_cpu,PASS\n");
  std::vector<std::string> algorithms = selected == "all"
      ? std::vector<std::string>{"nccl", "handwritten", "generated"}
      : std::vector<std::string>{selected};
  auto validate = [&]() {
    double max_error = 0;
    for (int r = 0; r < 4; ++r) {
      CUDA(cudaSetDevice(devs[r])); CUDA(cudaMemcpy(actual.data(), buffers[r], bytes, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < count; ++i) {
        if (!close_value(actual[i], reference[i])) {
          std::fprintf(stderr, "FAIL: logical rank %d (GPU %d), idx %zu, actual %.9g, expected %.9g\n", r, devs[r], i, actual[i], reference[i]);
          throw std::runtime_error("correctness mismatch");
        }
        max_error = std::max(max_error, std::abs(double(actual[i]) - reference[i]));
      }
    }
    return max_error;
  };
  for (const auto& algorithm : algorithms) {
    for (int w = 0; w < warmups; ++w) { reset(); run(algorithm, true); validate(); }
  }
  std::printf("kind,algorithm,bytes,iteration,delay_ms,physical_straggler,seed,runtime_ms,max_abs_error,correct\n");
  for (int iter = 0; iter < iterations; ++iter) {
    // Rotate ordering to reduce systematic warm-cache / clock drift bias.
    for (size_t a = 0; a < algorithms.size(); ++a) {
      const auto& algorithm = algorithms[(a + iter) % algorithms.size()];
      reset();
      auto start = std::chrono::steady_clock::now();
      run(algorithm, true);
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      double error = validate(); // outside the timed region, on every sample.
      std::printf("RESULT,%s,%zu,%d,%.6f,%d,%u,%.9f,%.9g,PASS\n",
                  algorithm.c_str(), bytes, iter, delay, straggler, seed, ms, error);
      std::fflush(stdout);
    }
  }
  for (int r = 0; r < 4; ++r) {
    CUDA(cudaSetDevice(devs[r]));
    CUDA(cudaFree(buffers[r])); CUDA(cudaFree(scratch[r]));
    NCCL(ncclCommDestroy(comms[r]));
    if (r < 3) NCCL(ncclCommDestroy(subcomms[r]));
    CUDA(cudaStreamDestroy(streams[r]));
    std::printf("Rank %d, done\n", r); // Nathnael's existing launcher contract.
  }
  std::printf("PASS: every sample on all four ranks matches ncclAllReduce\n");
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
}
