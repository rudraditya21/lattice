#include "lattice_kernel_common.h"

extern "C" __global__ void lattice_max_pool2d(const scalar_t* input, scalar_t* output,
                                              lattice_pool2d_params_t params) {
  unsigned long long col = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long row = static_cast<unsigned long long>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= params.out_h || col >= params.out_w) return;
  scalar_t best = -static_cast<scalar_t>(1e30);
  for (unsigned long long kr = 0; kr < params.k_h; ++kr) {
    for (unsigned long long kc = 0; kc < params.k_w; ++kc) {
      unsigned long long in_r = row * params.k_h + kr;
      unsigned long long in_c = col * params.k_w + kc;
      scalar_t val = input[in_r * params.in_w + in_c];
      if (val > best) best = val;
    }
  }
  output[row * params.out_w + col] = best;
}
