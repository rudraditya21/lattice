#include "lattice_kernel_common.h"

extern "C" __global__ void lattice_matmul(const scalar_t* a, const scalar_t* b, scalar_t* c,
                                          lattice_matmul_params_t params) {
  unsigned long long col = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long row = static_cast<unsigned long long>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= params.m || col >= params.n) return;
  scalar_t acc = static_cast<scalar_t>(0);
  for (unsigned long long k = 0; k < params.k; ++k) {
    acc += a[row * params.lda + k] * b[k * params.ldb + col];
  }
  c[row * params.ldc + col] = acc;
}
