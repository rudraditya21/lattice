#include "lattice_kernel_common.h"

kernel void lattice_conv2d(const device scalar_t* input [[buffer(0)]],
                           const device scalar_t* kernel [[buffer(1)]],
                           device scalar_t* output [[buffer(2)]],
                           constant lattice_conv2d_params_t& params [[buffer(3)]],
                           uint2 tid [[thread_position_in_grid]]) {
  ulong col = (ulong)tid.x;
  ulong row = (ulong)tid.y;
  if (row >= params.out_h || col >= params.out_w) return;
  scalar_t acc = (scalar_t)0;
  for (ulong kr = 0; kr < params.k_h; ++kr) {
    for (ulong kc = 0; kc < params.k_w; ++kc) {
      ulong in_r = row + kr;
      ulong in_c = col + kc;
      acc += input[in_r * params.in_w + in_c] * kernel[kr * params.k_w + kc];
    }
  }
  output[row * params.out_w + col] = acc;
}
