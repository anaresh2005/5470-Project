#include <initializer_list>

#include "validation.h"
#include <cassert>
#include <limits>
int main() {
  assert(!close_value(std::numeric_limits<float>::quiet_NaN(), 0));
  assert(!close_value(std::numeric_limits<float>::infinity(), 0));
  assert(!close_value(0, std::numeric_limits<float>::infinity()));
  assert(!close_value(1, 2));
  assert(close_value(1, 1.000001f));
  for (uint32_t seed : {0u, 5470u, UINT32_MAX}) {
    for (size_t i = 0; i < 4096; ++i) {
      float sum = 0;
      for (int rank = 0; rank < 4; ++rank) sum += input_value(rank, i, seed);
      assert(sum == expected_value(i, seed));
    }
  }
  assert(input_value(0, 0, 5470) != input_value(0, 1, 5470));
  assert(input_value(0, 0, 5470) != input_value(1, 0, 5470));
}
