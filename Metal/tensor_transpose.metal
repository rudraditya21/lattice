#include "lattice_kernel_common.h"

#define LATTICE_TRANSPOSE_TILE 16

kernel void lattice_transpose(const device scalar_t* input [[buffer(0)]],
                              device scalar_t* output [[buffer(1)]],
                              constant lattice_transpose_params_t& params
                              [[buffer(2)]],
                              uint2 tpos [[thread_position_in_threadgroup]],
                              uint2 gpos [[threadgroup_position_in_grid]]) {
    threadgroup scalar_t
        tile[LATTICE_TRANSPOSE_TILE][LATTICE_TRANSPOSE_TILE + 1];
    uint tx = tpos.x;
    uint ty = tpos.y;
    ulong col = (ulong)gpos.x * (ulong)LATTICE_TRANSPOSE_TILE + (ulong)tx;
    ulong row = (ulong)gpos.y * (ulong)LATTICE_TRANSPOSE_TILE + (ulong)ty;
    if (row < params.rows && col < params.cols) {
        tile[ty][tx] = input[row * params.cols + col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong out_row = (ulong)gpos.x * (ulong)LATTICE_TRANSPOSE_TILE + (ulong)ty;
    ulong out_col = (ulong)gpos.y * (ulong)LATTICE_TRANSPOSE_TILE + (ulong)tx;
    if (out_row < params.cols && out_col < params.rows) {
        output[out_row * params.rows + out_col] = tile[tx][ty];
    }
}

#undef LATTICE_TRANSPOSE_TILE
