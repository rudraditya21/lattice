#ifndef LATTICE_METAL_ABI_H
#define LATTICE_METAL_ABI_H

#ifndef LATTICE_ABI_VERSION
#define LATTICE_ABI_VERSION 0x00010001
#endif
#ifndef LATTICE_ABI_VERSION_MIN
#define LATTICE_ABI_VERSION_MIN 0x00010001
#endif

#define LATTICE_ABI_MAJOR(version) ((version) >> 16)
#define LATTICE_MAX_TENSOR_DIMS 8
#define LATTICE_ABI_STATIC_ASSERT(name, cond) typedef char name[(cond) ? 1 : -1]
#define LATTICE_ABI_OFFSETOF(type, field) __builtin_offsetof(type, field)
#define LATTICE_ABI_OFFSET_ASSERT(name, type, field, expected) \
  LATTICE_ABI_STATIC_ASSERT(name, LATTICE_ABI_OFFSETOF(type, field) == (expected))

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
  uint ndim;
  uint flags;
  ulong shape[LATTICE_MAX_TENSOR_DIMS];
  ulong out_strides[LATTICE_MAX_TENSOR_DIMS];
  ulong lhs_strides[LATTICE_MAX_TENSOR_DIMS];
  ulong rhs_strides[LATTICE_MAX_TENSOR_DIMS];
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

typedef struct lattice_transpose_params {
  ulong rows;
  ulong cols;
} lattice_transpose_params_t;

typedef struct lattice_conv2d_params {
  ulong in_h;
  ulong in_w;
  ulong k_h;
  ulong k_w;
  ulong out_h;
  ulong out_w;
} lattice_conv2d_params_t;

typedef struct lattice_pool2d_params {
  ulong in_h;
  ulong in_w;
  ulong k_h;
  ulong k_w;
  ulong out_h;
  ulong out_w;
} lattice_pool2d_params_t;

typedef struct lattice_fft_params {
  ulong n;
} lattice_fft_params_t;

typedef struct lattice_solve_params {
  ulong n;
  ulong rhs_cols;
} lattice_solve_params_t;

typedef struct lattice_lu_params {
  ulong n;
} lattice_lu_params_t;

typedef struct lattice_qr_params {
  ulong m;
  ulong n;
} lattice_qr_params_t;

typedef struct lattice_svd_params {
  ulong m;
  ulong n;
} lattice_svd_params_t;

typedef struct lattice_quantile_params {
  ulong count;
  float q;
  uint pad;
} lattice_quantile_params_t;

typedef struct lattice_correlation_params {
  ulong count;
} lattice_correlation_params_t;

typedef struct lattice_regression_params {
  ulong count;
} lattice_regression_params_t;

LATTICE_ABI_STATIC_ASSERT(lattice_abi_major_mismatch, LATTICE_ABI_MAJOR(LATTICE_ABI_VERSION) == 1);
LATTICE_ABI_STATIC_ASSERT(lattice_abi_version_too_old,
                          LATTICE_ABI_VERSION >= LATTICE_ABI_VERSION_MIN);
LATTICE_ABI_STATIC_ASSERT(lattice_elemwise_params_size, sizeof(lattice_elemwise_params_t) == 280);
LATTICE_ABI_OFFSET_ASSERT(lattice_elemwise_params_count_off, lattice_elemwise_params_t, count, 0);
LATTICE_ABI_OFFSET_ASSERT(lattice_elemwise_params_op_off, lattice_elemwise_params_t, op, 8);
LATTICE_ABI_OFFSET_ASSERT(lattice_elemwise_params_dtype_off, lattice_elemwise_params_t, dtype, 12);
LATTICE_ABI_OFFSET_ASSERT(lattice_elemwise_params_ndim_off, lattice_elemwise_params_t, ndim, 16);
LATTICE_ABI_OFFSET_ASSERT(lattice_elemwise_params_flags_off, lattice_elemwise_params_t, flags, 20);
LATTICE_ABI_OFFSET_ASSERT(lattice_elemwise_params_shape_off, lattice_elemwise_params_t, shape, 24);
LATTICE_ABI_OFFSET_ASSERT(lattice_elemwise_params_out_strides_off, lattice_elemwise_params_t,
                          out_strides, 24 + 8 * LATTICE_MAX_TENSOR_DIMS);
LATTICE_ABI_OFFSET_ASSERT(lattice_elemwise_params_lhs_strides_off, lattice_elemwise_params_t,
                          lhs_strides, 24 + 16 * LATTICE_MAX_TENSOR_DIMS);
LATTICE_ABI_OFFSET_ASSERT(lattice_elemwise_params_rhs_strides_off, lattice_elemwise_params_t,
                          rhs_strides, 24 + 24 * LATTICE_MAX_TENSOR_DIMS);
LATTICE_ABI_STATIC_ASSERT(lattice_reduce_params_size, sizeof(lattice_reduce_params_t) == 24);
LATTICE_ABI_OFFSET_ASSERT(lattice_reduce_params_count_off, lattice_reduce_params_t, count, 0);
LATTICE_ABI_OFFSET_ASSERT(lattice_reduce_params_op_off, lattice_reduce_params_t, op, 8);
LATTICE_ABI_OFFSET_ASSERT(lattice_reduce_params_dtype_off, lattice_reduce_params_t, dtype, 12);
LATTICE_ABI_OFFSET_ASSERT(lattice_reduce_params_stride_off, lattice_reduce_params_t, stride, 16);
LATTICE_ABI_STATIC_ASSERT(lattice_matmul_params_size, sizeof(lattice_matmul_params_t) == 56);
LATTICE_ABI_OFFSET_ASSERT(lattice_matmul_params_m_off, lattice_matmul_params_t, m, 0);
LATTICE_ABI_OFFSET_ASSERT(lattice_matmul_params_n_off, lattice_matmul_params_t, n, 8);
LATTICE_ABI_OFFSET_ASSERT(lattice_matmul_params_k_off, lattice_matmul_params_t, k, 16);
LATTICE_ABI_OFFSET_ASSERT(lattice_matmul_params_lda_off, lattice_matmul_params_t, lda, 24);
LATTICE_ABI_OFFSET_ASSERT(lattice_matmul_params_ldb_off, lattice_matmul_params_t, ldb, 32);
LATTICE_ABI_OFFSET_ASSERT(lattice_matmul_params_ldc_off, lattice_matmul_params_t, ldc, 40);
LATTICE_ABI_OFFSET_ASSERT(lattice_matmul_params_dtype_off, lattice_matmul_params_t, dtype, 48);
LATTICE_ABI_OFFSET_ASSERT(lattice_matmul_params_flags_off, lattice_matmul_params_t, flags, 52);
LATTICE_ABI_STATIC_ASSERT(lattice_transpose_params_size, sizeof(lattice_transpose_params_t) == 16);
LATTICE_ABI_STATIC_ASSERT(lattice_conv2d_params_size, sizeof(lattice_conv2d_params_t) == 48);
LATTICE_ABI_STATIC_ASSERT(lattice_pool2d_params_size, sizeof(lattice_pool2d_params_t) == 48);
LATTICE_ABI_STATIC_ASSERT(lattice_fft_params_size, sizeof(lattice_fft_params_t) == 8);
LATTICE_ABI_STATIC_ASSERT(lattice_solve_params_size, sizeof(lattice_solve_params_t) == 16);
LATTICE_ABI_STATIC_ASSERT(lattice_lu_params_size, sizeof(lattice_lu_params_t) == 8);
LATTICE_ABI_STATIC_ASSERT(lattice_qr_params_size, sizeof(lattice_qr_params_t) == 16);
LATTICE_ABI_STATIC_ASSERT(lattice_svd_params_size, sizeof(lattice_svd_params_t) == 16);
LATTICE_ABI_STATIC_ASSERT(lattice_quantile_params_size, sizeof(lattice_quantile_params_t) == 16);
LATTICE_ABI_STATIC_ASSERT(lattice_correlation_params_size,
                          sizeof(lattice_correlation_params_t) == 8);
LATTICE_ABI_STATIC_ASSERT(lattice_regression_params_size, sizeof(lattice_regression_params_t) == 8);

#endif  // LATTICE_METAL_ABI_H
