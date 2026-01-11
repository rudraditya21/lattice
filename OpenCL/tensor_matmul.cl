#include "lattice_kernel_common.h"

__kernel void lattice_matmul(__global const scalar_t* a, __global const scalar_t* b,
                             __global scalar_t* c, lattice_matmul_params_t params) {
  ulong col = (ulong)get_global_id(0);
  ulong row = (ulong)get_global_id(1);
  if (row >= params.m || col >= params.n) return;
  scalar_t acc = (scalar_t)0;
  for (ulong k = 0; k < params.k; ++k) {
    acc += a[row * params.lda + k] * b[k * params.ldb + col];
  }
  c[row * params.ldc + col] = acc;
}
