#include "lattice_kernel_common.h"

kernel void lattice_regression(const device scalar_t* x [[buffer(0)]],
                               const device scalar_t* y [[buffer(1)]],
                               device scalar_t* out [[buffer(2)]], device int* status [[buffer(3)]],
                               constant lattice_regression_params_t& params [[buffer(4)]],
                               uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  if (status) status[0] = 0;
  ulong count = params.count;
  scalar_t sumx = (scalar_t)0;
  scalar_t sumy = (scalar_t)0;
  scalar_t sumxx = (scalar_t)0;
  scalar_t sumxy = (scalar_t)0;
  for (ulong i = 0; i < count; ++i) {
    scalar_t xv = x[i];
    scalar_t yv = y[i];
    sumx += xv;
    sumy += yv;
    sumxx += xv * xv;
    sumxy += xv * yv;
  }
  scalar_t n = (scalar_t)count;
  scalar_t denom = n * sumxx - sumx * sumx;
  if (denom == (scalar_t)0) {
    if (status) status[0] = 1;
    out[0] = (scalar_t)0;
    out[1] = (scalar_t)0;
    return;
  }
  scalar_t slope = (n * sumxy - sumx * sumy) / denom;
  scalar_t intercept = (sumy - slope * sumx) / n;
  out[0] = slope;
  out[1] = intercept;
}
