#ifndef LATTICE_KERNEL_COMMON_H
#define LATTICE_KERNEL_COMMON_H

#include <metal_stdlib>

#include "lattice_abi.h"

using namespace metal;

#if defined(LATTICE_HAS_FP64) && (!defined(LATTICE_USE_FP64) || LATTICE_USE_FP64)
#define LATTICE_SCALAR_DOUBLE 1
using scalar_t = double;
#else
#define LATTICE_SCALAR_DOUBLE 0
using scalar_t = float;
#endif

#define LATTICE_REDUCE_MODE_SUM 1
#define LATTICE_REDUCE_MODE_MEAN 2
#define LATTICE_REDUCE_MODE_VAR 3
#define LATTICE_REDUCE_MODE_STD 4

inline ulong lattice_offset_from_index(ulong flat, const ulong* out_strides, const ulong* bstrides,
                                       uint ndim) {
  ulong offset = 0;
  ulong idx = flat;
  for (uint dim = 0; dim < ndim; ++dim) {
    ulong stride = out_strides[dim];
    ulong coord = stride == 0 ? 0 : (idx / stride);
    idx -= coord * stride;
    offset += coord * bstrides[dim];
  }
  return offset;
}

inline scalar_t lattice_abs(scalar_t v) {
#if LATTICE_SCALAR_DOUBLE
  return fabs(v);
#else
  return fabs(v);
#endif
}

inline scalar_t lattice_sqrt(scalar_t v) {
  return sqrt(v);
}
inline scalar_t lattice_cos(scalar_t v) {
  return cos(v);
}
inline scalar_t lattice_sin(scalar_t v) {
  return sin(v);
}

#endif  // LATTICE_KERNEL_COMMON_H
