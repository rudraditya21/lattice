#include "lattice_kernel_common.h"

#define LATTICE_KERNEL_NAME lattice_elemwise_div
#define LATTICE_VEC_KERNEL_NAME lattice_elemwise_div_vec4
#define LATTICE_ELEMWISE_OP(a, b) ((a) / (b))

#include "lattice_elemwise.inc"
