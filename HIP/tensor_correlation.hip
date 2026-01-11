#include "lattice_kernel_common.h"

extern "C" __global__ void lattice_correlation(const scalar_t* x, const scalar_t* y, scalar_t* out,
                                               int* status, lattice_correlation_params_t params) {
  unsigned long long tid = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid != 0) return;
  if (status) status[0] = 0;
  unsigned long long count = params.count;
  scalar_t sumx = static_cast<scalar_t>(0);
  scalar_t sumy = static_cast<scalar_t>(0);
  scalar_t sumxx = static_cast<scalar_t>(0);
  scalar_t sumyy = static_cast<scalar_t>(0);
  scalar_t sumxy = static_cast<scalar_t>(0);
  for (unsigned long long i = 0; i < count; ++i) {
    scalar_t xv = x[i];
    scalar_t yv = y[i];
    sumx += xv;
    sumy += yv;
    sumxx += xv * xv;
    sumyy += yv * yv;
    sumxy += xv * yv;
  }
  scalar_t n = static_cast<scalar_t>(count);
  scalar_t num = n * sumxy - sumx * sumy;
  scalar_t den = lattice_sqrt((n * sumxx - sumx * sumx) * (n * sumyy - sumy * sumy));
  if (den == static_cast<scalar_t>(0)) {
    if (status) status[0] = 1;
    out[0] = static_cast<scalar_t>(0);
    return;
  }
  out[0] = num / den;
}
