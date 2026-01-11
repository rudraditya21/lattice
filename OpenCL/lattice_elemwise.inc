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

#undef LATTICE_KERNEL_NAME
#undef LATTICE_ELEMWISE_OP
