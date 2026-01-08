#ifndef LATTICE_OPENCL_ABI_H
#define LATTICE_OPENCL_ABI_H

#ifndef LATTICE_ABI_VERSION
#define LATTICE_ABI_VERSION 0x00010000
#endif
#ifndef LATTICE_ABI_VERSION_MIN
#define LATTICE_ABI_VERSION_MIN 0x00010000
#endif

#define LATTICE_ABI_MAJOR(version) ((version) >> 16)
#define LATTICE_ABI_STATIC_ASSERT(name, cond) typedef char name[(cond) ? 1 : -1]
#if defined(__clang__) || defined(__GNUC__)
#define LATTICE_ABI_OFFSETOF(type, field) __builtin_offsetof(type, field)
#define LATTICE_ABI_OFFSET_ASSERT(name, type, field, expected) \
  LATTICE_ABI_STATIC_ASSERT(name, LATTICE_ABI_OFFSETOF(type, field) == (expected))
#else
#define LATTICE_ABI_OFFSET_ASSERT(name, type, field, expected)
#endif

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

LATTICE_ABI_STATIC_ASSERT(lattice_abi_major_mismatch,
                          LATTICE_ABI_MAJOR(LATTICE_ABI_VERSION) == 1);
LATTICE_ABI_STATIC_ASSERT(lattice_abi_version_too_old,
                          LATTICE_ABI_VERSION >= LATTICE_ABI_VERSION_MIN);
LATTICE_ABI_STATIC_ASSERT(lattice_elemwise_params_size, sizeof(lattice_elemwise_params_t) == 16);
LATTICE_ABI_OFFSET_ASSERT(lattice_elemwise_params_count_off, lattice_elemwise_params_t, count, 0);
LATTICE_ABI_OFFSET_ASSERT(lattice_elemwise_params_op_off, lattice_elemwise_params_t, op, 8);
LATTICE_ABI_OFFSET_ASSERT(lattice_elemwise_params_dtype_off, lattice_elemwise_params_t, dtype, 12);
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

#endif  // LATTICE_OPENCL_ABI_H
