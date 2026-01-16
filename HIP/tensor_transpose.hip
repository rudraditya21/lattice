#include "lattice_kernel_common.h"

#define LATTICE_TRANSPOSE_KERNEL(name, tile)                                 \
    extern "C" __global__ void name(const scalar_t* input, scalar_t* output, \
                                    lattice_transpose_params_t params) {     \
        __shared__ scalar_t tile_data[tile][tile + 1];                       \
        unsigned int tx = threadIdx.x;                                       \
        unsigned int ty = threadIdx.y;                                       \
        unsigned long long col =                                             \
            static_cast<unsigned long long>(blockIdx.x) * tile + tx;         \
        unsigned long long row =                                             \
            static_cast<unsigned long long>(blockIdx.y) * tile + ty;         \
        if (row < params.rows && col < params.cols) {                        \
            tile_data[ty][tx] = input[row * params.cols + col];              \
        }                                                                    \
        __syncthreads();                                                     \
        unsigned long long out_row =                                         \
            static_cast<unsigned long long>(blockIdx.x) * tile + ty;         \
        unsigned long long out_col =                                         \
            static_cast<unsigned long long>(blockIdx.y) * tile + tx;         \
        if (out_row < params.cols && out_col < params.rows) {                \
            output[out_row * params.rows + out_col] = tile_data[tx][ty];     \
        }                                                                    \
    }

LATTICE_TRANSPOSE_KERNEL(lattice_transpose, 16)
LATTICE_TRANSPOSE_KERNEL(lattice_transpose_t8, 8)

#undef LATTICE_TRANSPOSE_KERNEL
