#include "lattice_kernel_common.h"

#define LATTICE_MATMUL_KERNEL(name, tile)                                      \
    __kernel void name(__global const scalar_t* a, __global const scalar_t* b, \
                       __global scalar_t* c, lattice_matmul_params_t params) { \
        __local scalar_t As[tile][tile];                                       \
        __local scalar_t Bs[tile][tile];                                       \
        uint tx = get_local_id(0);                                             \
        uint ty = get_local_id(1);                                             \
        ulong row = (ulong)get_group_id(1) * (ulong)tile + (ulong)ty;          \
        ulong col = (ulong)get_group_id(0) * (ulong)tile + (ulong)tx;          \
        acc_t acc = (acc_t)0;                                                  \
        ulong tiles = (params.k + (ulong)tile - 1) / (ulong)tile;              \
        for (ulong t = 0; t < tiles; ++t) {                                    \
            ulong a_col = t * (ulong)tile + (ulong)tx;                         \
            ulong b_row = t * (ulong)tile + (ulong)ty;                         \
            As[ty][tx] = (row < params.m && a_col < params.k)                  \
                             ? a[row * params.lda + a_col]                     \
                             : (scalar_t)0;                                    \
            Bs[ty][tx] = (b_row < params.k && col < params.n)                  \
                             ? b[b_row * params.ldb + col]                     \
                             : (scalar_t)0;                                    \
            barrier(CLK_LOCAL_MEM_FENCE);                                      \
            for (uint k = 0; k < tile; ++k) {                                  \
                acc += (acc_t)As[ty][k] * (acc_t)Bs[k][tx];                    \
            }                                                                  \
            barrier(CLK_LOCAL_MEM_FENCE);                                      \
        }                                                                      \
        if (row < params.m && col < params.n) {                                \
            c[row * params.ldc + col] = (scalar_t)acc;                         \
        }                                                                      \
    }

LATTICE_MATMUL_KERNEL(lattice_matmul, 16)
LATTICE_MATMUL_KERNEL(lattice_matmul_t16, 16)
LATTICE_MATMUL_KERNEL(lattice_matmul_t32, 32)

#undef LATTICE_MATMUL_KERNEL
