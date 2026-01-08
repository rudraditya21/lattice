#ifndef LATTICE_HIP_ABI_H
#define LATTICE_HIP_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifndef LATTICE_ABI_VERSION
#define LATTICE_ABI_VERSION 0x00010000
#endif
#ifndef LATTICE_ABI_VERSION_MIN
#define LATTICE_ABI_VERSION_MIN 0x00010000
#endif

#define LATTICE_ABI_MAJOR(version) ((version) >> 16)

#if defined(__cplusplus)
#define LATTICE_ABI_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define LATTICE_ABI_STATIC_ASSERT(cond, msg) typedef char static_assert_##msg[(cond) ? 1 : -1]
#endif
#define LATTICE_ABI_OFFSET(type, field) offsetof(type, field)

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

LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_MAJOR(LATTICE_ABI_VERSION) == 1,
                          lattice_abi_major_mismatch);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_VERSION >= LATTICE_ABI_VERSION_MIN,
                          lattice_abi_version_too_old);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_elemwise_params_t) == 16,
                          lattice_elemwise_params_size);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_elemwise_params_t, count) == 0,
                          lattice_elemwise_params_count_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_elemwise_params_t, op) == 8,
                          lattice_elemwise_params_op_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_elemwise_params_t, dtype) == 12,
                          lattice_elemwise_params_dtype_off);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_reduce_params_t) == 24, lattice_reduce_params_size);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_reduce_params_t, count) == 0,
                          lattice_reduce_params_count_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_reduce_params_t, op) == 8,
                          lattice_reduce_params_op_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_reduce_params_t, dtype) == 12,
                          lattice_reduce_params_dtype_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_reduce_params_t, stride) == 16,
                          lattice_reduce_params_stride_off);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_matmul_params_t) == 56, lattice_matmul_params_size);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_matmul_params_t, m) == 0,
                          lattice_matmul_params_m_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_matmul_params_t, n) == 8,
                          lattice_matmul_params_n_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_matmul_params_t, k) == 16,
                          lattice_matmul_params_k_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_matmul_params_t, lda) == 24,
                          lattice_matmul_params_lda_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_matmul_params_t, ldb) == 32,
                          lattice_matmul_params_ldb_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_matmul_params_t, ldc) == 40,
                          lattice_matmul_params_ldc_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_matmul_params_t, dtype) == 48,
                          lattice_matmul_params_dtype_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_matmul_params_t, flags) == 52,
                          lattice_matmul_params_flags_off);

#endif  // LATTICE_HIP_ABI_H
