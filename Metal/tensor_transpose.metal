#include "lattice_kernel_common.h"

kernel void lattice_transpose(const device scalar_t* input [[buffer(0)]],
                              device scalar_t* output [[buffer(1)]],
                              constant lattice_transpose_params_t& params [[buffer(2)]],
                              uint2 tid [[thread_position_in_grid]]) {
  ulong col = (ulong)tid.x;
  ulong row = (ulong)tid.y;
  if (row >= params.rows || col >= params.cols) return;
  output[col * params.rows + row] = input[row * params.cols + col];
}
