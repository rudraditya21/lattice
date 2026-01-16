#ifndef LATTICE_KERNEL_COMMON_H
#define LATTICE_KERNEL_COMMON_H

#include <metal_stdlib>

#include "lattice_abi.h"

using namespace metal;

#if defined(LATTICE_HAS_FP64) && \
    (!defined(LATTICE_USE_FP64) || LATTICE_USE_FP64)
#define LATTICE_SCALAR_DOUBLE 1
using scalar_t = double;
#else
#define LATTICE_SCALAR_DOUBLE 0
using scalar_t = float;
#endif

#if defined(LATTICE_MIXED_PRECISION) && LATTICE_SCALAR_DOUBLE
using acc_t = float;
#else
using acc_t = scalar_t;
#endif

#define LATTICE_REDUCE_MODE_SUM 1
#define LATTICE_REDUCE_MODE_MEAN 2
#define LATTICE_REDUCE_MODE_VAR 3
#define LATTICE_REDUCE_MODE_STD 4

#if defined(LATTICE_VECTOR_WIDTH)
#define LATTICE_VEC_WIDTH LATTICE_VECTOR_WIDTH
#elif LATTICE_SCALAR_DOUBLE
#define LATTICE_VEC_WIDTH 2
#else
#define LATTICE_VEC_WIDTH 4
#endif

#if LATTICE_SCALAR_DOUBLE
using vec2_t = double2;
using vec4_t = double4;
#else
using vec2_t = float2;
using vec4_t = float4;
#endif

inline vec2_t lattice_vload2(const device scalar_t* ptr) {
    return *reinterpret_cast<const device vec2_t*>(ptr);
}

inline vec4_t lattice_vload4(const device scalar_t* ptr) {
    return *reinterpret_cast<const device vec4_t*>(ptr);
}

inline void lattice_vstore2(device scalar_t* ptr, vec2_t v) {
    *reinterpret_cast<device vec2_t*>(ptr) = v;
}

inline void lattice_vstore4(device scalar_t* ptr, vec4_t v) {
    *reinterpret_cast<device vec4_t*>(ptr) = v;
}

inline ulong lattice_offset_from_index(ulong flat,
                                       const ulong* out_strides,
                                       const ulong* bstrides,
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
#if defined(LATTICE_FAST_MATH)
    return fast::sqrt(v);
#else
    return sqrt(v);
#endif
}
inline scalar_t lattice_cos(scalar_t v) {
#if defined(LATTICE_FAST_MATH)
    return fast::cos(v);
#else
    return cos(v);
#endif
}
inline scalar_t lattice_sin(scalar_t v) {
#if defined(LATTICE_FAST_MATH)
    return fast::sin(v);
#else
    return sin(v);
#endif
}

#endif  // LATTICE_KERNEL_COMMON_H
