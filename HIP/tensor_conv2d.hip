#include "lattice_kernel_common.h"

extern "C" __global__ void lattice_conv2d(const scalar_t* input, const scalar_t* kernel,
                                          scalar_t* output, lattice_conv2d_params_t params) {
  unsigned long long col = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long row = static_cast<unsigned long long>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= params.out_h || col >= params.out_w) return;
  scalar_t acc = static_cast<scalar_t>(0);
  for (unsigned long long kr = 0; kr < params.k_h; ++kr) {
    for (unsigned long long kc = 0; kc < params.k_w; ++kc) {
      unsigned long long in_r = row + kr;
      unsigned long long in_c = col + kc;
      acc += input[in_r * params.in_w + in_c] * kernel[kr * params.k_w + kc];
    }
  }
  output[row * params.out_w + col] = acc;
}
