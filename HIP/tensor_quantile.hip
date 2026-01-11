#include "lattice_kernel_common.h"

extern "C" __global__ void lattice_quantile(const scalar_t* input, scalar_t* scratch, scalar_t* out,
                                            lattice_quantile_params_t params) {
  unsigned long long tid = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid != 0) return;
  unsigned long long count = params.count;
  if (count == 0) {
    out[0] = static_cast<scalar_t>(0);
    return;
  }
  for (unsigned long long i = 0; i < count; ++i) scratch[i] = input[i];
  for (unsigned long long i = 0; i < count; ++i) {
    unsigned long long min_idx = i;
    for (unsigned long long j = i + 1; j < count; ++j) {
      if (scratch[j] < scratch[min_idx]) min_idx = j;
    }
    if (min_idx != i) {
      scalar_t tmp = scratch[i];
      scratch[i] = scratch[min_idx];
      scratch[min_idx] = tmp;
    }
  }
  scalar_t q = static_cast<scalar_t>(params.q);
  scalar_t pos = q * static_cast<scalar_t>(count - 1);
  unsigned long long idx = static_cast<unsigned long long>(pos);
  scalar_t frac = pos - static_cast<scalar_t>(idx);
  scalar_t val = scratch[idx];
  if (idx + 1 < count) {
    val = val + frac * (scratch[idx + 1] - scratch[idx]);
  }
  out[0] = val;
}
