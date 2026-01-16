#include "lattice_kernel_common.h"

#define LATTICE_CONV2D_KERNEL(name, tile)                                     \
    extern "C" __global__ void name(const scalar_t* input,                    \
                                    const scalar_t* kernel, scalar_t* output, \
                                    lattice_conv2d_params_t params) {         \
        unsigned int tx = threadIdx.x;                                        \
        unsigned int ty = threadIdx.y;                                        \
        unsigned long long col =                                              \
            static_cast<unsigned long long>(blockIdx.x) * tile + tx;          \
        unsigned long long row =                                              \
            static_cast<unsigned long long>(blockIdx.y) * tile + ty;          \
        if (row >= params.out_h || col >= params.out_w)                       \
            return;                                                           \
        acc_t acc = static_cast<acc_t>(0);                                    \
        for (unsigned long long kr = 0; kr < params.k_h; ++kr) {              \
            for (unsigned long long kc = 0; kc < params.k_w; ++kc) {          \
                unsigned long long in_r = row + kr;                           \
                unsigned long long in_c = col + kc;                           \
                acc += static_cast<acc_t>(input[in_r * params.in_w + in_c]) * \
                       static_cast<acc_t>(kernel[kr * params.k_w + kc]);      \
            }                                                                 \
        }                                                                     \
        output[row * params.out_w + col] = static_cast<scalar_t>(acc);        \
    }

LATTICE_CONV2D_KERNEL(lattice_conv2d, 16)
LATTICE_CONV2D_KERNEL(lattice_conv2d_t8, 8)
LATTICE_CONV2D_KERNEL(lattice_conv2d_t16, 16)

#undef LATTICE_CONV2D_KERNEL
