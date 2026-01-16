#ifndef LATTICE_KERNEL_NAME
#error "LATTICE_KERNEL_NAME must be defined"
#endif
#ifndef LATTICE_ELEMWISE_OP
#error "LATTICE_ELEMWISE_OP must be defined"
#endif

__kernel void LATTICE_KERNEL_NAME(__global const scalar_t* lhs, __global const scalar_t* rhs,
                                  __global scalar_t* out,
                                  lattice_elemwise_params_t params) {
  ulong idx = (ulong)get_global_id(0);
  if (idx >= params.count) return;
  ulong lhs_off = lattice_offset_from_index(idx, params.out_strides, params.lhs_strides, params.ndim);
  ulong rhs_off = lattice_offset_from_index(idx, params.out_strides, params.rhs_strides, params.ndim);
  scalar_t a = lhs[lhs_off];
  scalar_t b = rhs[rhs_off];
  out[idx] = LATTICE_ELEMWISE_OP(a, b);
}

#ifdef LATTICE_VEC_KERNEL_NAME
__kernel void LATTICE_VEC_KERNEL_NAME(__global const scalar_t* lhs,
                                      __global const scalar_t* rhs,
                                      __global scalar_t* out,
                                      lattice_elemwise_params_t params) {
  ulong vec_idx = (ulong)get_global_id(0);
  ulong base = vec_idx * (ulong)LATTICE_VEC_WIDTH;
  if (base >= params.count) return;
#if LATTICE_VEC_WIDTH == 4
  if (base + 3 < params.count) {
    vec4_t a = lattice_vload4(lhs + base);
    vec4_t b = lattice_vload4(rhs + base);
    vec4_t c;
    c.s0 = LATTICE_ELEMWISE_OP(a.s0, b.s0);
    c.s1 = LATTICE_ELEMWISE_OP(a.s1, b.s1);
    c.s2 = LATTICE_ELEMWISE_OP(a.s2, b.s2);
    c.s3 = LATTICE_ELEMWISE_OP(a.s3, b.s3);
    lattice_vstore4(out + base, c);
  } else {
    for (ulong i = base; i < params.count; ++i) {
      out[i] = LATTICE_ELEMWISE_OP(lhs[i], rhs[i]);
    }
  }
#elif LATTICE_VEC_WIDTH == 2
  if (base + 1 < params.count) {
    vec2_t a = lattice_vload2(lhs + base);
    vec2_t b = lattice_vload2(rhs + base);
    vec2_t c;
    c.s0 = LATTICE_ELEMWISE_OP(a.s0, b.s0);
    c.s1 = LATTICE_ELEMWISE_OP(a.s1, b.s1);
    lattice_vstore2(out + base, c);
  } else {
    for (ulong i = base; i < params.count; ++i) {
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
