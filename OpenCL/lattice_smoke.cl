#include "lattice_abi.h"

__kernel void vec_add(__global const float* a, __global const float* b, __global float* out,
                      const uint n) {
  const uint gid = (uint)get_global_id(0);
  if (gid < n) {
    out[gid] = a[gid] + b[gid];
  }
}
