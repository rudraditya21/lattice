#include "lattice_kernel_common.h"

kernel void lattice_fft1d(const device scalar_t* input [[buffer(0)]],
                          device scalar_t* out_real [[buffer(1)]],
                          device scalar_t* out_imag [[buffer(2)]],
                          constant lattice_fft_params_t& params [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {
  ulong k = (ulong)tid;
  if (k >= params.n) return;
  scalar_t sum_r = (scalar_t)0;
  scalar_t sum_i = (scalar_t)0;
  const scalar_t two_pi = (scalar_t)6.28318530717958647692;
  for (ulong t = 0; t < params.n; ++t) {
    scalar_t angle = -two_pi * (scalar_t)(k * t) / (scalar_t)params.n;
    scalar_t val = input[t];
    sum_r += val * lattice_cos(angle);
    sum_i += val * lattice_sin(angle);
  }
  out_real[k] = sum_r;
  out_imag[k] = sum_i;
}
