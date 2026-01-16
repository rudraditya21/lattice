#include "lattice_kernel_common.h"

#define LATTICE_CONV2D_KERNEL(name, tile)                                \
    __kernel void name(                                                  \
        __global const scalar_t* input, __global const scalar_t* kernel, \
        __global scalar_t* output, lattice_conv2d_params_t params) {     \
        uint tx = get_local_id(0);                                       \
        uint ty = get_local_id(1);                                       \
        ulong col = (ulong)get_group_id(0) * (ulong)tile + (ulong)tx;    \
        ulong row = (ulong)get_group_id(1) * (ulong)tile + (ulong)ty;    \
        if (row >= params.out_h || col >= params.out_w)                  \
            return;                                                      \
        acc_t acc = (acc_t)0;                                            \
        for (ulong kr = 0; kr < params.k_h; ++kr) {                      \
            for (ulong kc = 0; kc < params.k_w; ++kc) {                  \
                ulong in_r = row + kr;                                   \
                ulong in_c = col + kc;                                   \
                acc += (acc_t)input[in_r * params.in_w + in_c] *         \
                       (acc_t)kernel[kr * params.k_w + kc];              \
            }                                                            \
        }                                                                \
        output[row * params.out_w + col] = (scalar_t)acc;                \
    }

LATTICE_CONV2D_KERNEL(lattice_conv2d, 16)
LATTICE_CONV2D_KERNEL(lattice_conv2d_t8, 8)
LATTICE_CONV2D_KERNEL(lattice_conv2d_t16, 16)

#undef LATTICE_CONV2D_KERNEL
