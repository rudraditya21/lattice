#include "lattice_kernel_common.h"

__kernel void lattice_qr(__global const scalar_t* a, __global scalar_t* q, __global scalar_t* r,
                         __global int* status, lattice_qr_params_t params) {
  ulong tid = (ulong)get_global_id(0);
  if (tid != 0) return;
  if (status) status[0] = 0;
  ulong m = params.m;
  ulong n = params.n;
  for (ulong i = 0; i < m * n; ++i) q[i] = a[i];
  for (ulong i = 0; i < n * n; ++i) r[i] = (scalar_t)0;
  for (ulong j = 0; j < n; ++j) {
    for (ulong k = 0; k < j; ++k) {
      scalar_t dot = (scalar_t)0;
      for (ulong i = 0; i < m; ++i) {
        dot += q[i * n + j] * q[i * n + k];
      }
      r[k * n + j] = dot;
      for (ulong i = 0; i < m; ++i) {
        q[i * n + j] -= dot * q[i * n + k];
      }
    }
    scalar_t norm = (scalar_t)0;
    for (ulong i = 0; i < m; ++i) {
      norm += q[i * n + j] * q[i * n + j];
    }
    norm = lattice_sqrt(norm);
    if (norm == (scalar_t)0) {
      if (status) status[0] = 1;
      return;
    }
    r[j * n + j] = norm;
    for (ulong i = 0; i < m; ++i) {
      q[i * n + j] /= norm;
    }
  }
}
