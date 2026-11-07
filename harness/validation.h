#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
// Rank- and index-dependent signed binary fractions. Integer-valued sums
// scaled by 1/256 are exactly representable, making indexing bugs visible.
HD inline float input_value(int physical_rank, size_t index, uint32_t seed) {
  uint32_t x = uint32_t(index) ^ (uint32_t(physical_rank + 1) * 0x9e3779b9u) ^ seed;
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
  return float(int(x % 2049u) - 1024) / 256.0f;
}
#undef HD
inline float expected_value(size_t index, uint32_t seed) {
  double sum = 0;
  for (int r = 0; r < 4; ++r) sum += input_value(r, index, seed);
  return static_cast<float>(sum);
}
inline bool close_value(float actual, float expected) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(double(actual) - expected) <= 1e-5 + 1e-5 * std::abs(double(expected));
}
