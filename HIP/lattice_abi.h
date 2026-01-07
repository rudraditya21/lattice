#ifndef LATTICE_HIP_ABI_H
#define LATTICE_HIP_ABI_H

#include <stdint.h>

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
  unsigned long long count;
  unsigned int op;
  unsigned int dtype;
} lattice_elemwise_params_t;

typedef struct lattice_reduce_params {
  unsigned long long count;
  unsigned int op;
  unsigned int dtype;
  unsigned long long stride;
} lattice_reduce_params_t;

typedef struct lattice_matmul_params {
  unsigned long long m;
  unsigned long long n;
  unsigned long long k;
  unsigned long long lda;
  unsigned long long ldb;
  unsigned long long ldc;
  unsigned int dtype;
  unsigned int flags;
} lattice_matmul_params_t;

#endif  // LATTICE_HIP_ABI_H
