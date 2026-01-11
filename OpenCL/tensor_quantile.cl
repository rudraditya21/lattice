#include "lattice_kernel_common.h"

__kernel void lattice_quantile(__global const scalar_t* input, __global scalar_t* scratch,
                               __global scalar_t* out, lattice_quantile_params_t params) {
  ulong tid = (ulong)get_global_id(0);
  if (tid != 0) return;
  ulong count = params.count;
  if (count == 0) {
    out[0] = (scalar_t)0;
    return;
  }
  for (ulong i = 0; i < count; ++i) scratch[i] = input[i];
  for (ulong i = 0; i < count; ++i) {
    ulong min_idx = i;
    for (ulong j = i + 1; j < count; ++j) {
      if (scratch[j] < scratch[min_idx]) min_idx = j;
    }
    if (min_idx != i) {
      scalar_t tmp = scratch[i];
      scratch[i] = scratch[min_idx];
      scratch[min_idx] = tmp;
    }
  }
  scalar_t q = (scalar_t)params.q;
  scalar_t pos = q * (scalar_t)(count - 1);
  ulong idx = (ulong)(pos);
  scalar_t frac = pos - (scalar_t)idx;
  scalar_t val = scratch[idx];
  if (idx + 1 < count) {
    val = val + frac * (scratch[idx + 1] - scratch[idx]);
  }
  out[0] = val;
}
