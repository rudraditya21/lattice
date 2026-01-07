#include "lattice_abi.h"

extern "C" __global__ void vec_add(const float* a, const float* b, float* out, unsigned int n) {
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    out[gid] = a[gid] + b[gid];
  }
}
