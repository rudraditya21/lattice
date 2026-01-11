#include "lattice_kernel_common.h"

extern "C" __global__ void lattice_regression(const scalar_t* x, const scalar_t* y, scalar_t* out,
                                              int* status, lattice_regression_params_t params) {
  unsigned long long tid = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid != 0) return;
  if (status) status[0] = 0;
  unsigned long long count = params.count;
  scalar_t sumx = static_cast<scalar_t>(0);
  scalar_t sumy = static_cast<scalar_t>(0);
  scalar_t sumxx = static_cast<scalar_t>(0);
  scalar_t sumxy = static_cast<scalar_t>(0);
  for (unsigned long long i = 0; i < count; ++i) {
    scalar_t xv = x[i];
    scalar_t yv = y[i];
    sumx += xv;
    sumy += yv;
    sumxx += xv * xv;
    sumxy += xv * yv;
  }
  scalar_t n = static_cast<scalar_t>(count);
  scalar_t denom = n * sumxx - sumx * sumx;
  if (denom == static_cast<scalar_t>(0)) {
    if (status) status[0] = 1;
    out[0] = static_cast<scalar_t>(0);
    out[1] = static_cast<scalar_t>(0);
    return;
  }
  scalar_t slope = (n * sumxy - sumx * sumy) / denom;
  scalar_t intercept = (sumy - slope * sumx) / n;
  out[0] = slope;
  out[1] = intercept;
}
