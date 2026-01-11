#include "lattice_kernel_common.h"

kernel void lattice_matmul(const device scalar_t* a [[buffer(0)]],
                           const device scalar_t* b [[buffer(1)]], device scalar_t* c [[buffer(2)]],
                           constant lattice_matmul_params_t& params [[buffer(3)]],
                           uint2 tid [[thread_position_in_grid]]) {
  ulong col = (ulong)tid.x;
  ulong row = (ulong)tid.y;
  if (row >= params.m || col >= params.n) return;
  scalar_t acc = (scalar_t)0;
  for (ulong k = 0; k < params.k; ++k) {
    acc += a[row * params.lda + k] * b[k * params.ldb + col];
  }
  c[row * params.ldc + col] = acc;
}
