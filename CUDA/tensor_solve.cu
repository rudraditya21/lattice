#include "lattice_kernel_common.h"

extern "C" __global__ void lattice_solve(const scalar_t* a, const scalar_t* b, scalar_t* out,
                                         scalar_t* scratch, int* status,
                                         lattice_solve_params_t params) {
  unsigned long long tid = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid != 0) return;
  if (status) status[0] = 0;
  unsigned long long n = params.n;
  unsigned long long rhs_cols = params.rhs_cols;
  unsigned long long a_size = n * n;
  unsigned long long b_size = n * rhs_cols;
  for (unsigned long long i = 0; i < a_size; ++i) scratch[i] = a[i];
  for (unsigned long long i = 0; i < b_size; ++i) out[i] = b[i];
  for (unsigned long long k = 0; k < n; ++k) {
    unsigned long long pivot = k;
    scalar_t maxv = lattice_abs(scratch[k * n + k]);
    for (unsigned long long i = k + 1; i < n; ++i) {
      scalar_t v = lattice_abs(scratch[i * n + k]);
      if (v > maxv) {
        maxv = v;
        pivot = i;
      }
    }
    if (maxv == static_cast<scalar_t>(0)) {
      if (status) status[0] = 1;
      return;
    }
    if (pivot != k) {
      for (unsigned long long j = 0; j < n; ++j) {
        scalar_t tmp = scratch[k * n + j];
        scratch[k * n + j] = scratch[pivot * n + j];
        scratch[pivot * n + j] = tmp;
      }
      for (unsigned long long j = 0; j < rhs_cols; ++j) {
        scalar_t tmp = out[k * rhs_cols + j];
        out[k * rhs_cols + j] = out[pivot * rhs_cols + j];
        out[pivot * rhs_cols + j] = tmp;
      }
    }
    scalar_t diag = scratch[k * n + k];
    for (unsigned long long j = k; j < n; ++j) scratch[k * n + j] /= diag;
    for (unsigned long long j = 0; j < rhs_cols; ++j) out[k * rhs_cols + j] /= diag;
    for (unsigned long long i = 0; i < n; ++i) {
      if (i == k) continue;
      scalar_t factor = scratch[i * n + k];
      for (unsigned long long j = k; j < n; ++j) {
        scratch[i * n + j] -= factor * scratch[k * n + j];
      }
      for (unsigned long long j = 0; j < rhs_cols; ++j) {
        out[i * rhs_cols + j] -= factor * out[k * rhs_cols + j];
      }
    }
  }
}
