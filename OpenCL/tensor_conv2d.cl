#include "lattice_kernel_common.h"

__kernel void lattice_conv2d(__global const scalar_t* input, __global const scalar_t* kernel,
                             __global scalar_t* output, lattice_conv2d_params_t params) {
  ulong col = (ulong)get_global_id(0);
  ulong row = (ulong)get_global_id(1);
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
