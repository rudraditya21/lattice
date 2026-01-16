#ifndef LATTICE_KERNEL_NAME
#error "LATTICE_KERNEL_NAME must be defined"
#endif
#ifndef LATTICE_PARAM_BUFFER_INDEX
#error "LATTICE_PARAM_BUFFER_INDEX must be defined"
#endif
#ifndef LATTICE_ELEMWISE_OP
#error "LATTICE_ELEMWISE_OP must be defined"
#endif

kernel void LATTICE_KERNEL_NAME(const device scalar_t* lhs [[buffer(0)]],
                                const device scalar_t* rhs [[buffer(1)]],
                                device scalar_t* out [[buffer(2)]],
                                constant lattice_elemwise_params_t& params
                                    [[buffer(LATTICE_PARAM_BUFFER_INDEX)]],
                                uint tid [[thread_position_in_grid]]) {
  ulong idx = (ulong)tid;
  if (idx >= params.count) return;
  ulong lhs_off = lattice_offset_from_index(idx, params.out_strides, params.lhs_strides, params.ndim);
  ulong rhs_off = lattice_offset_from_index(idx, params.out_strides, params.rhs_strides, params.ndim);
  scalar_t a = lhs[lhs_off];
  scalar_t b = rhs[rhs_off];
  out[idx] = LATTICE_ELEMWISE_OP(a, b);
}

#ifdef LATTICE_VEC_KERNEL_NAME
kernel void LATTICE_VEC_KERNEL_NAME(const device scalar_t* lhs [[buffer(0)]],
                                    const device scalar_t* rhs [[buffer(1)]],
                                    device scalar_t* out [[buffer(2)]],
                                    constant lattice_elemwise_params_t& params
                                        [[buffer(LATTICE_PARAM_BUFFER_INDEX)]],
                                    uint tid [[thread_position_in_grid]]) {
  ulong vec_idx = (ulong)tid;
  ulong base = vec_idx * (ulong)LATTICE_VEC_WIDTH;
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
    for (ulong i = base; i < params.count; ++i) {
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
#undef LATTICE_PARAM_BUFFER_INDEX
#undef LATTICE_ELEMWISE_OP
#undef LATTICE_VEC_KERNEL_NAME
