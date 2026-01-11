#include "lattice_kernel_common.h"

#define LATTICE_KERNEL_NAME lattice_reduce_var
#define LATTICE_PARAM_BUFFER_INDEX 2
#define LATTICE_REDUCE_MODE LATTICE_REDUCE_MODE_VAR

#include "lattice_reduce.inc"
