#include "lattice_kernel_common.h"

#define LATTICE_KERNEL_NAME lattice_reduce_mean
#define LATTICE_REDUCE_MODE LATTICE_REDUCE_MODE_MEAN

#include "lattice_reduce.inc"
