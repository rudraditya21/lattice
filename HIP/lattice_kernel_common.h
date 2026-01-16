#ifndef LATTICE_KERNEL_COMMON_H
#define LATTICE_KERNEL_COMMON_H

#include <math.h>

#include "lattice_abi.h"

#if defined(LATTICE_HAS_FP64) && \
    (!defined(LATTICE_USE_FP64) || LATTICE_USE_FP64)
#define LATTICE_SCALAR_DOUBLE 1
typedef double scalar_t;
#else
#define LATTICE_SCALAR_DOUBLE 0
typedef float scalar_t;
#endif

#if defined(LATTICE_MIXED_PRECISION) && LATTICE_SCALAR_DOUBLE
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
#elif LATTICE_SCALAR_DOUBLE
#define LATTICE_VEC_WIDTH 2
#else
#define LATTICE_VEC_WIDTH 4
#endif

#if LATTICE_SCALAR_DOUBLE
typedef double2 vec2_t;
typedef double4 vec4_t;
#else
typedef float2 vec2_t;
typedef float4 vec4_t;
#endif

__device__ __forceinline__ vec2_t lattice_vload2(const scalar_t* ptr) {
    return *reinterpret_cast<const vec2_t*>(ptr);
}

__device__ __forceinline__ vec4_t lattice_vload4(const scalar_t* ptr) {
    return *reinterpret_cast<const vec4_t*>(ptr);
}

__device__ __forceinline__ void lattice_vstore2(scalar_t* ptr, vec2_t v) {
    *reinterpret_cast<vec2_t*>(ptr) = v;
}

__device__ __forceinline__ void lattice_vstore4(scalar_t* ptr, vec4_t v) {
    *reinterpret_cast<vec4_t*>(ptr) = v;
}

__device__ __forceinline__ unsigned long long lattice_offset_from_index(
    unsigned long long flat,
    const unsigned long long* out_strides,
    const unsigned long long* bstrides,
    unsigned int ndim) {
    unsigned long long offset = 0;
    unsigned long long idx = flat;
    for (unsigned int dim = 0; dim < ndim; ++dim) {
        unsigned long long stride = out_strides[dim];
        unsigned long long coord = stride == 0 ? 0 : (idx / stride);
        idx -= coord * stride;
        offset += coord * bstrides[dim];
    }
    return offset;
}

__device__ __forceinline__ scalar_t lattice_abs(scalar_t v) {
#if LATTICE_SCALAR_DOUBLE
    return fabs(v);
#else
    return fabsf(v);
#endif
}

__device__ __forceinline__ scalar_t lattice_sqrt(scalar_t v) {
#if LATTICE_SCALAR_DOUBLE
    return sqrt(v);
#elif defined(LATTICE_FAST_MATH)
    return __fsqrt_rn(v);
#else
    return sqrtf(v);
#endif
}

__device__ __forceinline__ scalar_t lattice_cos(scalar_t v) {
#if LATTICE_SCALAR_DOUBLE
    return cos(v);
#elif defined(LATTICE_FAST_MATH)
    return __cosf(v);
#else
    return cosf(v);
#endif
}

__device__ __forceinline__ scalar_t lattice_sin(scalar_t v) {
#if LATTICE_SCALAR_DOUBLE
    return sin(v);
#elif defined(LATTICE_FAST_MATH)
    return __sinf(v);
#else
    return sinf(v);
#endif
}

#endif  // LATTICE_KERNEL_COMMON_H
