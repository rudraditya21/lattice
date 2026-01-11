#include "lattice_kernel_common.h"

__kernel void lattice_max_pool2d(__global const scalar_t* input, __global scalar_t* output,
                                 lattice_pool2d_params_t params) {
  ulong col = (ulong)get_global_id(0);
  ulong row = (ulong)get_global_id(1);
  if (row >= params.out_h || col >= params.out_w) return;
  scalar_t best = (scalar_t)(-1e30f);
  for (ulong kr = 0; kr < params.k_h; ++kr) {
    for (ulong kc = 0; kc < params.k_w; ++kc) {
      ulong in_r = row * params.k_h + kr;
      ulong in_c = col * params.k_w + kc;
      scalar_t val = input[in_r * params.in_w + in_c];
      if (val > best) best = val;
    }
  }
  output[row * params.out_w + col] = best;
}
