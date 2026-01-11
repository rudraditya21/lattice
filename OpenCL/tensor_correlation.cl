#include "lattice_kernel_common.h"

__kernel void lattice_correlation(__global const scalar_t* x, __global const scalar_t* y,
                                  __global scalar_t* out, __global int* status,
                                  lattice_correlation_params_t params) {
  ulong tid = (ulong)get_global_id(0);
  if (tid != 0) return;
  if (status) status[0] = 0;
  ulong count = params.count;
  scalar_t sumx = (scalar_t)0;
  scalar_t sumy = (scalar_t)0;
  scalar_t sumxx = (scalar_t)0;
  scalar_t sumyy = (scalar_t)0;
  scalar_t sumxy = (scalar_t)0;
  for (ulong i = 0; i < count; ++i) {
    scalar_t xv = x[i];
    scalar_t yv = y[i];
    sumx += xv;
    sumy += yv;
    sumxx += xv * xv;
    sumyy += yv * yv;
    sumxy += xv * yv;
  }
  scalar_t n = (scalar_t)count;
  scalar_t num = n * sumxy - sumx * sumy;
  scalar_t den = lattice_sqrt((n * sumxx - sumx * sumx) * (n * sumyy - sumy * sumy));
  if (den == (scalar_t)0) {
    if (status) status[0] = 1;
    out[0] = (scalar_t)0;
    return;
  }
  out[0] = num / den;
}
