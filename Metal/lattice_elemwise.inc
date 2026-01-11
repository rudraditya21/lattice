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

#undef LATTICE_KERNEL_NAME
#undef LATTICE_PARAM_BUFFER_INDEX
#undef LATTICE_ELEMWISE_OP
