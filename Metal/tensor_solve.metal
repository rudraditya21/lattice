#include "lattice_kernel_common.h"

kernel void lattice_solve(const device scalar_t* a [[buffer(0)]],
                          const device scalar_t* b [[buffer(1)]],
                          device scalar_t* out [[buffer(2)]],
                          device scalar_t* scratch [[buffer(3)]], device int* status [[buffer(4)]],
                          constant lattice_solve_params_t& params [[buffer(5)]],
                          uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  if (status) status[0] = 0;
  ulong n = params.n;
  ulong rhs_cols = params.rhs_cols;
  ulong a_size = n * n;
  ulong b_size = n * rhs_cols;
  for (ulong i = 0; i < a_size; ++i) scratch[i] = a[i];
  for (ulong i = 0; i < b_size; ++i) out[i] = b[i];
  for (ulong k = 0; k < n; ++k) {
    ulong pivot = k;
    scalar_t maxv = lattice_abs(scratch[k * n + k]);
    for (ulong i = k + 1; i < n; ++i) {
      scalar_t v = lattice_abs(scratch[i * n + k]);
      if (v > maxv) {
        maxv = v;
        pivot = i;
      }
    }
    if (maxv == (scalar_t)0) {
      if (status) status[0] = 1;
      return;
    }
    if (pivot != k) {
      for (ulong j = 0; j < n; ++j) {
        scalar_t tmp = scratch[k * n + j];
        scratch[k * n + j] = scratch[pivot * n + j];
        scratch[pivot * n + j] = tmp;
      }
      for (ulong j = 0; j < rhs_cols; ++j) {
        scalar_t tmp = out[k * rhs_cols + j];
        out[k * rhs_cols + j] = out[pivot * rhs_cols + j];
        out[pivot * rhs_cols + j] = tmp;
      }
    }
    scalar_t diag = scratch[k * n + k];
    for (ulong j = k; j < n; ++j) scratch[k * n + j] /= diag;
    for (ulong j = 0; j < rhs_cols; ++j) out[k * rhs_cols + j] /= diag;
    for (ulong i = 0; i < n; ++i) {
      if (i == k) continue;
      scalar_t factor = scratch[i * n + k];
      for (ulong j = k; j < n; ++j) {
        scratch[i * n + j] -= factor * scratch[k * n + j];
      }
      for (ulong j = 0; j < rhs_cols; ++j) {
        out[i * rhs_cols + j] -= factor * out[k * rhs_cols + j];
      }
    }
  }
}
