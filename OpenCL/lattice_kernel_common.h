#ifndef LATTICE_KERNEL_COMMON_H
#define LATTICE_KERNEL_COMMON_H

#include "lattice_abi.h"

#if defined(LATTICE_HAS_FP64) && \
    (!defined(LATTICE_USE_FP64) || LATTICE_USE_FP64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double scalar_t;
#else
typedef float scalar_t;
#endif

#if defined(LATTICE_MIXED_PRECISION) && defined(LATTICE_HAS_FP64) && \
    (!defined(LATTICE_USE_FP64) || LATTICE_USE_FP64)
typedef float acc_t;
#else
typedef scalar_t acc_t;
#endif

#define LATTICE_REDUCE_MODE_SUM 1
#define LATTICE_REDUCE_MODE_MEAN 2
#define LATTICE_REDUCE_MODE_VAR 3
#define LATTICE_REDUCE_MODE_STD 4

#if defined(LATTICE_VECTOR_WIDTH)
#define LATTICE_VEC_WIDTH LATTICE_VECTOR_WIDTH
#elif defined(LATTICE_HAS_FP64) && \
    (!defined(LATTICE_USE_FP64) || LATTICE_USE_FP64)
#define LATTICE_VEC_WIDTH 2
#else
#define LATTICE_VEC_WIDTH 4
#endif

#if defined(LATTICE_HAS_FP64) && \
    (!defined(LATTICE_USE_FP64) || LATTICE_USE_FP64)
typedef double2 vec2_t;
typedef double4 vec4_t;
#else
typedef float2 vec2_t;
typedef float4 vec4_t;
#endif

static inline vec2_t lattice_vload2(const scalar_t* ptr) {
    return vload2(0, ptr);
}

static inline vec4_t lattice_vload4(const scalar_t* ptr) {
    return vload4(0, ptr);
}

static inline void lattice_vstore2(scalar_t* ptr, vec2_t v) {
    vstore2(v, 0, ptr);
}

static inline void lattice_vstore4(scalar_t* ptr, vec4_t v) {
    vstore4(v, 0, ptr);
}

static inline ulong lattice_offset_from_index(ulong flat,
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

static inline scalar_t lattice_abs(scalar_t v) {
    return fabs(v);
}
static inline scalar_t lattice_sqrt(scalar_t v) {
#if defined(LATTICE_FAST_MATH)
    return native_sqrt(v);
#else
    return sqrt(v);
#endif
}
static inline scalar_t lattice_cos(scalar_t v) {
#if defined(LATTICE_FAST_MATH)
    return native_cos(v);
#else
    return cos(v);
#endif
}
static inline scalar_t lattice_sin(scalar_t v) {
#if defined(LATTICE_FAST_MATH)
    return native_sin(v);
#else
    return sin(v);
#endif
}

#endif  // LATTICE_KERNEL_COMMON_H
