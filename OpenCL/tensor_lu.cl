#include "lattice_kernel_common.h"

__kernel void lattice_lu(__global const scalar_t* a, __global scalar_t* l, __global scalar_t* u,
                         __global int* status, lattice_lu_params_t params) {
  ulong tid = (ulong)get_global_id(0);
  if (tid != 0) return;
  if (status) status[0] = 0;
  ulong n = params.n;
  for (ulong i = 0; i < n * n; ++i) {
    l[i] = (scalar_t)0;
    u[i] = (scalar_t)0;
  }
  for (ulong i = 0; i < n; ++i) {
    l[i * n + i] = (scalar_t)1;
  }
  for (ulong k = 0; k < n; ++k) {
    for (ulong j = k; j < n; ++j) {
      scalar_t sum = (scalar_t)0;
      for (ulong s = 0; s < k; ++s) {
        sum += l[k * n + s] * u[s * n + j];
      }
      u[k * n + j] = a[k * n + j] - sum;
    }
    if (u[k * n + k] == (scalar_t)0) {
      if (status) status[0] = 1;
      return;
    }
    for (ulong i = k + 1; i < n; ++i) {
      scalar_t sum = (scalar_t)0;
      for (ulong s = 0; s < k; ++s) {
        sum += l[i * n + s] * u[s * n + k];
      }
      l[i * n + k] = (a[i * n + k] - sum) / u[k * n + k];
    }
  }
}
