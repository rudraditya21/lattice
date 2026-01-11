#ifndef LATTICE_KERNEL_NAME
#error "LATTICE_KERNEL_NAME must be defined"
#endif
#ifndef LATTICE_REDUCE_MODE
#error "LATTICE_REDUCE_MODE must be defined"
#endif

extern "C" __global__ void LATTICE_KERNEL_NAME(const scalar_t* input, scalar_t* out,
                                                lattice_reduce_params_t params) {
  unsigned long long tid =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid != 0) return;
  unsigned long long count = params.count;
  if (count == 0) {
    out[0] = static_cast<scalar_t>(0);
    return;
  }
  scalar_t total = static_cast<scalar_t>(0);
  for (unsigned long long i = 0; i < count; ++i) {
    total += input[i];
  }
#if LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_SUM
  out[0] = total;
#elif LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_MEAN
  out[0] = total / static_cast<scalar_t>(count);
#elif LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_VAR || LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_STD
  scalar_t mean = total / static_cast<scalar_t>(count);
  scalar_t sum_sq = static_cast<scalar_t>(0);
  for (unsigned long long i = 0; i < count; ++i) {
    scalar_t diff = input[i] - mean;
    sum_sq += diff * diff;
  }
  scalar_t variance = sum_sq / static_cast<scalar_t>(count);
#if LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_VAR
  out[0] = variance;
#else
  out[0] = lattice_sqrt(variance);
#endif
#else
  out[0] = total;
#endif
}

#undef LATTICE_KERNEL_NAME
#undef LATTICE_REDUCE_MODE
