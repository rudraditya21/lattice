#include <metal_stdlib>

#include "lattice_abi.h"

using namespace metal;

kernel void vec_add(const device float* a [[buffer(0)]], const device float* b [[buffer(1)]],
                    device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
  if (gid < n) {
    out[gid] = a[gid] + b[gid];
  }
}
