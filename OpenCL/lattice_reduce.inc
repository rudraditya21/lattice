#ifndef LATTICE_KERNEL_NAME
#error "LATTICE_KERNEL_NAME must be defined"
#endif
#ifndef LATTICE_REDUCE_MODE
#error "LATTICE_REDUCE_MODE must be defined"
#endif

__kernel void LATTICE_KERNEL_NAME(__global const scalar_t* input, __global scalar_t* out,
                                  lattice_reduce_params_t params) {
#ifndef LATTICE_REDUCE_BLOCK
#define LATTICE_REDUCE_BLOCK 256
#endif
  __local acc_t scratch[LATTICE_REDUCE_BLOCK];
  uint tid = get_local_id(0);
  uint local_size = get_local_size(0);
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
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = local_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
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
  barrier(CLK_LOCAL_MEM_FENCE);
  acc_t mean = scratch[0];
  acc_t sum_sq = (acc_t)0;
  for (ulong i = (ulong)tid; i < count; i += (ulong)local_size) {
    acc_t diff = (acc_t)input[i] - mean;
    sum_sq += diff * diff;
  }
  scratch[tid] = sum_sq;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint stride = local_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
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
#undef LATTICE_REDUCE_MODE
