#include "lattice_kernel_common.h"

#define LATTICE_TRANSPOSE_KERNEL(name, tile)                             \
    kernel void name(const device scalar_t* input [[buffer(0)]],         \
                     device scalar_t* output [[buffer(1)]],              \
                     constant lattice_transpose_params_t& params         \
                     [[buffer(2)]],                                      \
                     uint2 tpos [[thread_position_in_threadgroup]],      \
                     uint2 gpos [[threadgroup_position_in_grid]]) {      \
        threadgroup scalar_t tile_data[tile][tile + 1];                  \
        uint tx = tpos.x;                                                \
        uint ty = tpos.y;                                                \
        ulong col = (ulong)gpos.x * (ulong)tile + (ulong)tx;             \
        ulong row = (ulong)gpos.y * (ulong)tile + (ulong)ty;             \
        if (row < params.rows && col < params.cols) {                    \
            tile_data[ty][tx] = input[row * params.cols + col];          \
        }                                                                \
        threadgroup_barrier(mem_flags::mem_threadgroup);                 \
        ulong out_row = (ulong)gpos.x * (ulong)tile + (ulong)ty;         \
        ulong out_col = (ulong)gpos.y * (ulong)tile + (ulong)tx;         \
        if (out_row < params.cols && out_col < params.rows) {            \
            output[out_row * params.rows + out_col] = tile_data[tx][ty]; \
        }                                                                \
    }

LATTICE_TRANSPOSE_KERNEL(lattice_transpose, 16)
LATTICE_TRANSPOSE_KERNEL(lattice_transpose_t8, 8)

#undef LATTICE_TRANSPOSE_KERNEL
