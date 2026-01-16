#include "lattice_kernel_common.h"

#define LATTICE_MATMUL_KERNEL(name, tile)                                    \
    kernel void name(const device scalar_t* a [[buffer(0)]],                 \
                     const device scalar_t* b [[buffer(1)]],                 \
                     device scalar_t* c [[buffer(2)]],                       \
                     constant lattice_matmul_params_t& params [[buffer(3)]], \
                     uint2 tpos [[thread_position_in_threadgroup]],          \
                     uint2 gpos [[threadgroup_position_in_grid]]) {          \
        threadgroup scalar_t As[tile][tile];                                 \
        threadgroup scalar_t Bs[tile][tile];                                 \
        uint tx = tpos.x;                                                    \
        uint ty = tpos.y;                                                    \
        ulong row = (ulong)gpos.y * (ulong)tile + (ulong)ty;                 \
        ulong col = (ulong)gpos.x * (ulong)tile + (ulong)tx;                 \
        acc_t acc = (acc_t)0;                                                \
        ulong tiles = (params.k + (ulong)tile - 1) / (ulong)tile;            \
        for (ulong t = 0; t < tiles; ++t) {                                  \
            ulong a_col = t * (ulong)tile + (ulong)tx;                       \
            ulong b_row = t * (ulong)tile + (ulong)ty;                       \
            As[ty][tx] = (row < params.m && a_col < params.k)                \
                             ? a[row * params.lda + a_col]                   \
                             : (scalar_t)0;                                  \
            Bs[ty][tx] = (b_row < params.k && col < params.n)                \
                             ? b[b_row * params.ldb + col]                   \
                             : (scalar_t)0;                                  \
            threadgroup_barrier(mem_flags::mem_threadgroup);                 \
            for (uint k = 0; k < tile; ++k) {                                \
                acc += (acc_t)As[ty][k] * (acc_t)Bs[k][tx];                  \
            }                                                                \
            threadgroup_barrier(mem_flags::mem_threadgroup);                 \
        }                                                                    \
        if (row < params.m && col < params.n) {                              \
            c[row * params.ldc + col] = (scalar_t)acc;                       \
        }                                                                    \
    }

LATTICE_MATMUL_KERNEL(lattice_matmul, 16)
LATTICE_MATMUL_KERNEL(lattice_matmul_t16, 16)
LATTICE_MATMUL_KERNEL(lattice_matmul_t32, 32)

#undef LATTICE_MATMUL_KERNEL
