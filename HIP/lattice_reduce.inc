#ifndef LATTICE_KERNEL_NAME
#error "LATTICE_KERNEL_NAME must be defined"
#endif
#ifndef LATTICE_REDUCE_MODE
#error "LATTICE_REDUCE_MODE must be defined"
#endif

extern "C" __global__ void LATTICE_KERNEL_NAME(const scalar_t* input, scalar_t* out,
                                                lattice_reduce_params_t params) {
#ifndef LATTICE_REDUCE_BLOCK
#define LATTICE_REDUCE_BLOCK 256
#endif
  __shared__ acc_t scratch[LATTICE_REDUCE_BLOCK];
  unsigned int tid = threadIdx.x;
  unsigned long long count = params.count;
  if (count == 0) {
    if (tid == 0) {
      out[0] = static_cast<scalar_t>(0);
    }
    return;
  }
  acc_t sum = static_cast<acc_t>(0);
  for (unsigned long long i = tid; i < count; i += blockDim.x) {
    sum += static_cast<acc_t>(input[i]);
  }
  scratch[tid] = sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    acc_t total = scratch[0];
#if LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_SUM
    out[0] = static_cast<scalar_t>(total);
#elif LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_MEAN
    out[0] = static_cast<scalar_t>(total / static_cast<acc_t>(count));
#elif LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_VAR || LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_STD
    scratch[0] = total / static_cast<acc_t>(count);
#else
    out[0] = static_cast<scalar_t>(total);
#endif
  }
#if LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_VAR || LATTICE_REDUCE_MODE == LATTICE_REDUCE_MODE_STD
  __syncthreads();
  acc_t mean = scratch[0];
  acc_t sum_sq = static_cast<acc_t>(0);
  for (unsigned long long i = tid; i < count; i += blockDim.x) {
    acc_t diff = static_cast<acc_t>(input[i]) - mean;
    sum_sq += diff * diff;
  }
  scratch[tid] = sum_sq;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    acc_t variance = scratch[0] / static_cast<acc_t>(count);
    scalar_t out_val = static_cast<scalar_t>(variance);
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
