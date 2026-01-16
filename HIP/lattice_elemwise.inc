#ifndef LATTICE_KERNEL_NAME
#error "LATTICE_KERNEL_NAME must be defined"
#endif
#ifndef LATTICE_ELEMWISE_OP
#error "LATTICE_ELEMWISE_OP must be defined"
#endif

extern "C" __global__ void LATTICE_KERNEL_NAME(const scalar_t* lhs, const scalar_t* rhs,
                                                scalar_t* out,
                                                lattice_elemwise_params_t params) {
  unsigned long long idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= params.count) return;
  unsigned long long lhs_off =
      lattice_offset_from_index(idx, params.out_strides, params.lhs_strides, params.ndim);
  unsigned long long rhs_off =
      lattice_offset_from_index(idx, params.out_strides, params.rhs_strides, params.ndim);
  scalar_t a = lhs[lhs_off];
  scalar_t b = rhs[rhs_off];
  out[idx] = LATTICE_ELEMWISE_OP(a, b);
}

#ifdef LATTICE_VEC_KERNEL_NAME
extern "C" __global__ void LATTICE_VEC_KERNEL_NAME(const scalar_t* lhs,
                                                   const scalar_t* rhs,
                                                   scalar_t* out,
                                                   lattice_elemwise_params_t params) {
  unsigned long long vec_idx =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long base = vec_idx * LATTICE_VEC_WIDTH;
  if (base >= params.count) return;
#if LATTICE_VEC_WIDTH == 4
  if (base + 3 < params.count) {
    vec4_t a = lattice_vload4(lhs + base);
    vec4_t b = lattice_vload4(rhs + base);
    vec4_t c;
    c.x = LATTICE_ELEMWISE_OP(a.x, b.x);
    c.y = LATTICE_ELEMWISE_OP(a.y, b.y);
    c.z = LATTICE_ELEMWISE_OP(a.z, b.z);
    c.w = LATTICE_ELEMWISE_OP(a.w, b.w);
    lattice_vstore4(out + base, c);
  } else {
    for (unsigned long long i = base; i < params.count; ++i) {
      out[i] = LATTICE_ELEMWISE_OP(lhs[i], rhs[i]);
    }
  }
#elif LATTICE_VEC_WIDTH == 2
  if (base + 1 < params.count) {
    vec2_t a = lattice_vload2(lhs + base);
    vec2_t b = lattice_vload2(rhs + base);
    vec2_t c;
    c.x = LATTICE_ELEMWISE_OP(a.x, b.x);
    c.y = LATTICE_ELEMWISE_OP(a.y, b.y);
    lattice_vstore2(out + base, c);
  } else {
    for (unsigned long long i = base; i < params.count; ++i) {
      out[i] = LATTICE_ELEMWISE_OP(lhs[i], rhs[i]);
    }
  }
#else
  out[base] = LATTICE_ELEMWISE_OP(lhs[base], rhs[base]);
#endif
}
#endif

#undef LATTICE_KERNEL_NAME
#undef LATTICE_ELEMWISE_OP
#undef LATTICE_VEC_KERNEL_NAME
