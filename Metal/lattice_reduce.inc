#ifndef LATTICE_KERNEL_NAME
#error "LATTICE_KERNEL_NAME must be defined"
#endif
#ifndef LATTICE_PARAM_BUFFER_INDEX
#error "LATTICE_PARAM_BUFFER_INDEX must be defined"
#endif
#ifndef LATTICE_REDUCE_MODE
#error "LATTICE_REDUCE_MODE must be defined"
#endif

kernel void LATTICE_KERNEL_NAME(const device scalar_t* input [[buffer(0)]],
                                device scalar_t* out [[buffer(1)]],
                                constant lattice_reduce_params_t& params
                                    [[buffer(LATTICE_PARAM_BUFFER_INDEX)]],
                                uint tid [[thread_position_in_grid]]) {
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
#undef LATTICE_PARAM_BUFFER_INDEX
#undef LATTICE_REDUCE_MODE
