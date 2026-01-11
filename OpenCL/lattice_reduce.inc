#ifndef LATTICE_KERNEL_NAME
#error "LATTICE_KERNEL_NAME must be defined"
#endif
#ifndef LATTICE_REDUCE_MODE
#error "LATTICE_REDUCE_MODE must be defined"
#endif

__kernel void LATTICE_KERNEL_NAME(__global const scalar_t* input, __global scalar_t* out,
                                  lattice_reduce_params_t params) {
  ulong tid = (ulong)get_global_id(0);
  if (tid != 0) return;
  ulong count = params.count;
  if (count == 0) {
    out[0] = (scalar_t)0;
    return;
  }
  scalar_t total = (scalar_t)0;
  for (ulong i = 0; i < count; ++i) {
    total += input[i];
  }
#if LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_SUM
  out[0] = total;
#elif LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_MEAN
  out[0] = total / (scalar_t)count;
#elif LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_VAR || LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_STD
  scalar_t mean = total / (scalar_t)count;
  scalar_t sum_sq = (scalar_t)0;
  for (ulong i = 0; i < count; ++i) {
    scalar_t diff = input[i] - mean;
    sum_sq += diff * diff;
  }
  scalar_t variance = sum_sq / (scalar_t)count;
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
