#ifndef LATTICE_HIP_ABI_H
#define LATTICE_HIP_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifndef LATTICE_ABI_VERSION
#define LATTICE_ABI_VERSION 0x00010001
#endif
#ifndef LATTICE_ABI_VERSION_MIN
#define LATTICE_ABI_VERSION_MIN 0x00010001
#endif

#define LATTICE_ABI_MAJOR(version) ((version) >> 16)
#define LATTICE_MAX_TENSOR_DIMS 8

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
  unsigned int ndim;
  unsigned int flags;
  unsigned long long shape[LATTICE_MAX_TENSOR_DIMS];
  unsigned long long out_strides[LATTICE_MAX_TENSOR_DIMS];
  unsigned long long lhs_strides[LATTICE_MAX_TENSOR_DIMS];
  unsigned long long rhs_strides[LATTICE_MAX_TENSOR_DIMS];
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

typedef struct lattice_transpose_params {
  unsigned long long rows;
  unsigned long long cols;
} lattice_transpose_params_t;

typedef struct lattice_conv2d_params {
  unsigned long long in_h;
  unsigned long long in_w;
  unsigned long long k_h;
  unsigned long long k_w;
  unsigned long long out_h;
  unsigned long long out_w;
} lattice_conv2d_params_t;

typedef struct lattice_pool2d_params {
  unsigned long long in_h;
  unsigned long long in_w;
  unsigned long long k_h;
  unsigned long long k_w;
  unsigned long long out_h;
  unsigned long long out_w;
} lattice_pool2d_params_t;

typedef struct lattice_fft_params {
  unsigned long long n;
} lattice_fft_params_t;

typedef struct lattice_solve_params {
  unsigned long long n;
  unsigned long long rhs_cols;
} lattice_solve_params_t;

typedef struct lattice_lu_params {
  unsigned long long n;
} lattice_lu_params_t;

typedef struct lattice_qr_params {
  unsigned long long m;
  unsigned long long n;
} lattice_qr_params_t;

typedef struct lattice_svd_params {
  unsigned long long m;
  unsigned long long n;
} lattice_svd_params_t;

typedef struct lattice_quantile_params {
  unsigned long long count;
  float q;
  unsigned int pad;
} lattice_quantile_params_t;

typedef struct lattice_correlation_params {
  unsigned long long count;
} lattice_correlation_params_t;

typedef struct lattice_regression_params {
  unsigned long long count;
} lattice_regression_params_t;

LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_MAJOR(LATTICE_ABI_VERSION) == 1, lattice_abi_major_mismatch);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_VERSION >= LATTICE_ABI_VERSION_MIN,
                          lattice_abi_version_too_old);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_elemwise_params_t) == 280, lattice_elemwise_params_size);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_elemwise_params_t, count) == 0,
                          lattice_elemwise_params_count_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_elemwise_params_t, op) == 8,
                          lattice_elemwise_params_op_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_elemwise_params_t, dtype) == 12,
                          lattice_elemwise_params_dtype_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_elemwise_params_t, ndim) == 16,
                          lattice_elemwise_params_ndim_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_elemwise_params_t, flags) == 20,
                          lattice_elemwise_params_flags_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_elemwise_params_t, shape) == 24,
                          lattice_elemwise_params_shape_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_elemwise_params_t, out_strides) ==
                              24 + 8 * LATTICE_MAX_TENSOR_DIMS,
                          lattice_elemwise_params_out_strides_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_elemwise_params_t, lhs_strides) ==
                              24 + 16 * LATTICE_MAX_TENSOR_DIMS,
                          lattice_elemwise_params_lhs_strides_off);
LATTICE_ABI_STATIC_ASSERT(LATTICE_ABI_OFFSET(lattice_elemwise_params_t, rhs_strides) ==
                              24 + 24 * LATTICE_MAX_TENSOR_DIMS,
                          lattice_elemwise_params_rhs_strides_off);
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
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_transpose_params_t) == 16, lattice_transpose_params_size);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_conv2d_params_t) == 48, lattice_conv2d_params_size);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_pool2d_params_t) == 48, lattice_pool2d_params_size);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_fft_params_t) == 8, lattice_fft_params_size);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_solve_params_t) == 16, lattice_solve_params_size);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_lu_params_t) == 8, lattice_lu_params_size);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_qr_params_t) == 16, lattice_qr_params_size);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_svd_params_t) == 16, lattice_svd_params_size);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_quantile_params_t) == 16, lattice_quantile_params_size);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_correlation_params_t) == 8,
                          lattice_correlation_params_size);
LATTICE_ABI_STATIC_ASSERT(sizeof(lattice_regression_params_t) == 8, lattice_regression_params_size);

#endif  // LATTICE_HIP_ABI_H
