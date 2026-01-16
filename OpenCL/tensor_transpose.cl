#include "lattice_kernel_common.h"

#define LATTICE_TRANSPOSE_KERNEL(name, tile)                              \
    __kernel void name(__global const scalar_t* input,                    \
                       __global scalar_t* output,                         \
                       lattice_transpose_params_t params) {               \
        __local scalar_t tile_data[tile][tile + 1];                       \
        uint tx = get_local_id(0);                                        \
        uint ty = get_local_id(1);                                        \
        ulong col = (ulong)get_group_id(0) * (ulong)tile + (ulong)tx;     \
        ulong row = (ulong)get_group_id(1) * (ulong)tile + (ulong)ty;     \
        if (row < params.rows && col < params.cols) {                     \
            tile_data[ty][tx] = input[row * params.cols + col];           \
        }                                                                 \
        barrier(CLK_LOCAL_MEM_FENCE);                                     \
        ulong out_row = (ulong)get_group_id(0) * (ulong)tile + (ulong)ty; \
        ulong out_col = (ulong)get_group_id(1) * (ulong)tile + (ulong)tx; \
        if (out_row < params.cols && out_col < params.rows) {             \
            output[out_row * params.rows + out_col] = tile_data[tx][ty];  \
        }                                                                 \
    }

LATTICE_TRANSPOSE_KERNEL(lattice_transpose, 16)
LATTICE_TRANSPOSE_KERNEL(lattice_transpose_t8, 8)

#undef LATTICE_TRANSPOSE_KERNEL
