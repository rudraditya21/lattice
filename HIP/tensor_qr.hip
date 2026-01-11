#include "lattice_kernel_common.h"

extern "C" __global__ void lattice_qr(const scalar_t* a, scalar_t* q, scalar_t* r, int* status,
                                      lattice_qr_params_t params) {
  unsigned long long tid = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid != 0) return;
  if (status) status[0] = 0;
  unsigned long long m = params.m;
  unsigned long long n = params.n;
  for (unsigned long long i = 0; i < m * n; ++i) q[i] = a[i];
  for (unsigned long long i = 0; i < n * n; ++i) r[i] = static_cast<scalar_t>(0);
  for (unsigned long long j = 0; j < n; ++j) {
    for (unsigned long long k = 0; k < j; ++k) {
      scalar_t dot = static_cast<scalar_t>(0);
      for (unsigned long long i = 0; i < m; ++i) {
        dot += q[i * n + j] * q[i * n + k];
      }
      r[k * n + j] = dot;
      for (unsigned long long i = 0; i < m; ++i) {
        q[i * n + j] -= dot * q[i * n + k];
      }
    }
    scalar_t norm = static_cast<scalar_t>(0);
    for (unsigned long long i = 0; i < m; ++i) {
      norm += q[i * n + j] * q[i * n + j];
    }
    norm = lattice_sqrt(norm);
    if (norm == static_cast<scalar_t>(0)) {
      if (status) status[0] = 1;
      return;
    }
    r[j * n + j] = norm;
    for (unsigned long long i = 0; i < m; ++i) {
      q[i * n + j] /= norm;
    }
  }
}
