#include "lattice_kernel_common.h"

#define LATTICE_CONV2D_KERNEL(name, tile)                                    \
    kernel void name(const device scalar_t* input [[buffer(0)]],             \
                     const device scalar_t* kernel [[buffer(1)]],            \
                     device scalar_t* output [[buffer(2)]],                  \
                     constant lattice_conv2d_params_t& params [[buffer(3)]], \
                     uint2 tpos [[thread_position_in_threadgroup]],          \
                     uint2 gpos [[threadgroup_position_in_grid]]) {          \
        uint tx = tpos.x;                                                    \
        uint ty = tpos.y;                                                    \
        ulong col = (ulong)gpos.x * (ulong)tile + (ulong)tx;                 \
        ulong row = (ulong)gpos.y * (ulong)tile + (ulong)ty;                 \
        if (row >= params.out_h || col >= params.out_w)                      \
            return;                                                          \
        acc_t acc = (acc_t)0;                                                \
        for (ulong kr = 0; kr < params.k_h; ++kr) {                          \
            for (ulong kc = 0; kc < params.k_w; ++kc) {                      \
                ulong in_r = row + kr;                                       \
                ulong in_c = col + kc;                                       \
                acc += (acc_t)input[in_r * params.in_w + in_c] *             \
                       (acc_t)kernel[kr * params.k_w + kc];                  \
            }                                                                \
        }                                                                    \
        output[row * params.out_w + col] = (scalar_t)acc;                    \
    }

LATTICE_CONV2D_KERNEL(lattice_conv2d, 16)
LATTICE_CONV2D_KERNEL(lattice_conv2d_t8, 8)
LATTICE_CONV2D_KERNEL(lattice_conv2d_t16, 16)

#undef LATTICE_CONV2D_KERNEL
