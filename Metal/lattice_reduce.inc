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
                                uint tid [[thread_index_in_threadgroup]],
                                uint3 tpg [[threads_per_threadgroup]]) {
#ifndef LATTICE_REDUCE_BLOCK
#define LATTICE_REDUCE_BLOCK 256
#endif
  threadgroup acc_t scratch[LATTICE_REDUCE_BLOCK];
  uint local_size = tpg.x;
  ulong count = params.count;
  if (count == 0) {
    if (tid == 0) {
      out[0] = (scalar_t)0;
    }
    return;
  }
  acc_t sum = (acc_t)0;
  for (ulong i = (ulong)tid; i < count; i += (ulong)local_size) {
    sum += (acc_t)input[i];
  }
  scratch[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = local_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    acc_t total = scratch[0];
#if LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_SUM
    out[0] = (scalar_t)total;
#elif LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_MEAN
    out[0] = (scalar_t)(total / (acc_t)count);
#elif LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_VAR || LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_STD
    scratch[0] = total / (acc_t)count;
#else
    out[0] = (scalar_t)total;
#endif
  }
#if LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_VAR || LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_STD
  threadgroup_barrier(mem_flags::mem_threadgroup);
  acc_t mean = scratch[0];
  acc_t sum_sq = (acc_t)0;
  for (ulong i = (ulong)tid; i < count; i += (ulong)local_size) {
    acc_t diff = (acc_t)input[i] - mean;
    sum_sq += diff * diff;
  }
  scratch[tid] = sum_sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = local_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    acc_t variance = scratch[0] / (acc_t)count;
    scalar_t out_val = (scalar_t)variance;
#if LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_VAR
    out[0] = out_val;
#else
    out[0] = lattice_sqrt(out_val);
#endif
  }
#endif
}

#undef LATTICE_KERNEL_NAME
#undef LATTICE_PARAM_BUFFER_INDEX
#undef LATTICE_REDUCE_MODE
