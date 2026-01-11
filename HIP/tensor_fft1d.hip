#include "lattice_kernel_common.h"

extern "C" __global__ void lattice_fft1d(const scalar_t* input, scalar_t* out_real,
                                         scalar_t* out_imag, lattice_fft_params_t params) {
  unsigned long long k = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= params.n) return;
  scalar_t sum_r = static_cast<scalar_t>(0);
  scalar_t sum_i = static_cast<scalar_t>(0);
  const scalar_t two_pi = static_cast<scalar_t>(6.28318530717958647692);
  for (unsigned long long t = 0; t < params.n; ++t) {
    scalar_t angle = -two_pi * static_cast<scalar_t>(k * t) / static_cast<scalar_t>(params.n);
    scalar_t val = input[t];
    sum_r += val * lattice_cos(angle);
    sum_i += val * lattice_sin(angle);
  }
  out_real[k] = sum_r;
  out_imag[k] = sum_i;
}
