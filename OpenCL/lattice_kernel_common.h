#ifndef LATTICE_KERNEL_COMMON_H
#define LATTICE_KERNEL_COMMON_H

#include "lattice_abi.h"

#if defined(LATTICE_HAS_FP64) && (!defined(LATTICE_USE_FP64) || LATTICE_USE_FP64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double scalar_t;
#else
typedef float scalar_t;
#endif

#define LATTICE_REDUCE_MODE_SUM 1
#define LATTICE_REDUCE_MODE_MEAN 2
#define LATTICE_REDUCE_MODE_VAR 3
#define LATTICE_REDUCE_MODE_STD 4

static inline ulong lattice_offset_from_index(ulong flat, const ulong* out_strides,
                                              const ulong* bstrides, uint ndim) {
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

static inline scalar_t lattice_abs(scalar_t v) {
  return fabs(v);
}
static inline scalar_t lattice_sqrt(scalar_t v) {
  return sqrt(v);
}
static inline scalar_t lattice_cos(scalar_t v) {
  return cos(v);
}
static inline scalar_t lattice_sin(scalar_t v) {
  return sin(v);
}

#endif  // LATTICE_KERNEL_COMMON_H
