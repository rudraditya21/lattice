#include "lattice_kernel_common.h"

extern "C" __global__ void lattice_transpose(const scalar_t* input, scalar_t* output,
                                             lattice_transpose_params_t params) {
  unsigned long long col = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long row = static_cast<unsigned long long>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (row >= params.rows || col >= params.cols) return;
  output[col * params.rows + row] = input[row * params.cols + col];
}
