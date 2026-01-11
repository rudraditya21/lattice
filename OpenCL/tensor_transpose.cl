#include "lattice_kernel_common.h"

__kernel void lattice_transpose(__global const scalar_t* input, __global scalar_t* output,
                                lattice_transpose_params_t params) {
  ulong col = (ulong)get_global_id(0);
  ulong row = (ulong)get_global_id(1);
  if (row >= params.rows || col >= params.cols) return;
  output[col * params.rows + row] = input[row * params.cols + col];
}
