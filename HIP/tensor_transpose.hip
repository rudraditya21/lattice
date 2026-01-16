#include "lattice_kernel_common.h"

#define LATTICE_TRANSPOSE_TILE 16

extern "C" __global__ void lattice_transpose(
    const scalar_t* input,
    scalar_t* output,
    lattice_transpose_params_t params) {
    __shared__ scalar_t
        tile[LATTICE_TRANSPOSE_TILE][LATTICE_TRANSPOSE_TILE + 1];
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned long long col =
        static_cast<unsigned long long>(blockIdx.x) * LATTICE_TRANSPOSE_TILE +
        tx;
    unsigned long long row =
        static_cast<unsigned long long>(blockIdx.y) * LATTICE_TRANSPOSE_TILE +
        ty;
    if (row < params.rows && col < params.cols) {
        tile[ty][tx] = input[row * params.cols + col];
    }
    __syncthreads();
    unsigned long long out_row =
        static_cast<unsigned long long>(blockIdx.x) * LATTICE_TRANSPOSE_TILE +
        ty;
    unsigned long long out_col =
        static_cast<unsigned long long>(blockIdx.y) * LATTICE_TRANSPOSE_TILE +
        tx;
    if (out_row < params.cols && out_col < params.rows) {
        output[out_row * params.rows + out_col] = tile[tx][ty];
    }
}

#undef LATTICE_TRANSPOSE_TILE
