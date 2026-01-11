#include "lattice_kernel_common.h"

kernel void lattice_max_pool2d(const device scalar_t* input [[buffer(0)]],
                               device scalar_t* output [[buffer(1)]],
                               constant lattice_pool2d_params_t& params [[buffer(2)]],
                               uint2 tid [[thread_position_in_grid]]) {
  ulong col = (ulong)tid.x;
  ulong row = (ulong)tid.y;
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
