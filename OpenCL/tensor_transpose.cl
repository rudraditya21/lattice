#include "lattice_kernel_common.h"

#define LATTICE_TRANSPOSE_TILE 16

__kernel void lattice_transpose(__global const scalar_t* input,
                                __global scalar_t* output,
                                lattice_transpose_params_t params) {
    __local scalar_t tile[LATTICE_TRANSPOSE_TILE][LATTICE_TRANSPOSE_TILE + 1];
    uint tx = get_local_id(0);
    uint ty = get_local_id(1);
    ulong col =
        (ulong)get_group_id(0) * (ulong)LATTICE_TRANSPOSE_TILE + (ulong)tx;
    ulong row =
        (ulong)get_group_id(1) * (ulong)LATTICE_TRANSPOSE_TILE + (ulong)ty;
    if (row < params.rows && col < params.cols) {
        tile[ty][tx] = input[row * params.cols + col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ulong out_row =
        (ulong)get_group_id(0) * (ulong)LATTICE_TRANSPOSE_TILE + (ulong)ty;
    ulong out_col =
        (ulong)get_group_id(1) * (ulong)LATTICE_TRANSPOSE_TILE + (ulong)tx;
    if (out_row < params.cols && out_col < params.rows) {
        output[out_row * params.rows + out_col] = tile[tx][ty];
    }
}

#undef LATTICE_TRANSPOSE_TILE
