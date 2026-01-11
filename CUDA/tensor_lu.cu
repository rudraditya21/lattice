#include "lattice_kernel_common.h"

extern "C" __global__ void lattice_lu(const scalar_t* a, scalar_t* l, scalar_t* u, int* status,
                                      lattice_lu_params_t params) {
  unsigned long long tid = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid != 0) return;
  if (status) status[0] = 0;
  unsigned long long n = params.n;
  for (unsigned long long i = 0; i < n * n; ++i) {
    l[i] = static_cast<scalar_t>(0);
    u[i] = static_cast<scalar_t>(0);
  }
  for (unsigned long long i = 0; i < n; ++i) {
    l[i * n + i] = static_cast<scalar_t>(1);
  }
  for (unsigned long long k = 0; k < n; ++k) {
    for (unsigned long long j = k; j < n; ++j) {
      scalar_t sum = static_cast<scalar_t>(0);
      for (unsigned long long s = 0; s < k; ++s) {
        sum += l[k * n + s] * u[s * n + j];
      }
      u[k * n + j] = a[k * n + j] - sum;
    }
    if (u[k * n + k] == static_cast<scalar_t>(0)) {
      if (status) status[0] = 1;
      return;
    }
    for (unsigned long long i = k + 1; i < n; ++i) {
      scalar_t sum = static_cast<scalar_t>(0);
      for (unsigned long long s = 0; s < k; ++s) {
        sum += l[i * n + s] * u[s * n + k];
      }
      l[i * n + k] = (a[i * n + k] - sum) / u[k * n + k];
    }
  }
}
