#include "lattice_kernel_common.h"

#define LATTICE_MATMUL_KERNEL(name, tile)                                  \
    extern "C" __global__ void name(const scalar_t* a, const scalar_t* b,  \
                                    scalar_t* c,                           \
                                    lattice_matmul_params_t params) {      \
        __shared__ scalar_t As[tile][tile];                                \
        __shared__ scalar_t Bs[tile][tile];                                \
        unsigned int tx = threadIdx.x;                                     \
        unsigned int ty = threadIdx.y;                                     \
        unsigned long long row =                                           \
            static_cast<unsigned long long>(blockIdx.y) * tile + ty;       \
        unsigned long long col =                                           \
            static_cast<unsigned long long>(blockIdx.x) * tile + tx;       \
        acc_t acc = static_cast<acc_t>(0);                                 \
        unsigned long long tiles =                                         \
            (params.k + static_cast<unsigned long long>(tile) - 1) / tile; \
        for (unsigned long long t = 0; t < tiles; ++t) {                   \
            unsigned long long a_col =                                     \
                t * tile + static_cast<unsigned long long>(tx);            \
            unsigned long long b_row =                                     \
                t * tile + static_cast<unsigned long long>(ty);            \
            As[ty][tx] = (row < params.m && a_col < params.k)              \
                             ? a[row * params.lda + a_col]                 \
                             : static_cast<scalar_t>(0);                   \
            Bs[ty][tx] = (b_row < params.k && col < params.n)              \
                             ? b[b_row * params.ldb + col]                 \
                             : static_cast<scalar_t>(0);                   \
            __syncthreads();                                               \
            for (unsigned int k = 0; k < tile; ++k) {                      \
                acc += static_cast<acc_t>(As[ty][k]) *                     \
                       static_cast<acc_t>(Bs[k][tx]);                      \
            }                                                              \
            __syncthreads();                                               \
        }                                                                  \
        if (row < params.m && col < params.n) {                            \
            c[row * params.ldc + col] = static_cast<scalar_t>(acc);        \
        }                                                                  \
    }

LATTICE_MATMUL_KERNEL(lattice_matmul, 16)
LATTICE_MATMUL_KERNEL(lattice_matmul_t16, 16)
LATTICE_MATMUL_KERNEL(lattice_matmul_t32, 32)
LATTICE_MATMUL_KERNEL(lattice_matmul_t8, 8)

#undef LATTICE_MATMUL_KERNEL
