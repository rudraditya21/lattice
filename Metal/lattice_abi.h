#ifndef LATTICE_METAL_ABI_H
#define LATTICE_METAL_ABI_H

#define LATTICE_ABI_VERSION 1

typedef enum lattice_elemwise_op {
  LATTICE_ELEMWISE_ADD = 1,
  LATTICE_ELEMWISE_SUB = 2,
  LATTICE_ELEMWISE_MUL = 3,
  LATTICE_ELEMWISE_DIV = 4,
} lattice_elemwise_op_t;

typedef enum lattice_reduce_op {
  LATTICE_REDUCE_SUM = 1,
  LATTICE_REDUCE_MIN = 2,
  LATTICE_REDUCE_MAX = 3,
} lattice_reduce_op_t;

typedef enum lattice_dtype_code {
  LATTICE_DTYPE_F32 = 1,
  LATTICE_DTYPE_F64 = 2,
  LATTICE_DTYPE_I32 = 3,
  LATTICE_DTYPE_U32 = 4,
} lattice_dtype_code_t;

// Fixed ABI: kernels receive input buffers first, then a params struct by value.
typedef struct lattice_elemwise_params {
  ulong count;
  uint op;
  uint dtype;
} lattice_elemwise_params_t;

typedef struct lattice_reduce_params {
  ulong count;
  uint op;
  uint dtype;
  ulong stride;
} lattice_reduce_params_t;

typedef struct lattice_matmul_params {
  ulong m;
  ulong n;
  ulong k;
  ulong lda;
  ulong ldb;
  ulong ldc;
  uint dtype;
  uint flags;
} lattice_matmul_params_t;

#endif  // LATTICE_METAL_ABI_H
