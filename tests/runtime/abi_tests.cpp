#include "test_util.h"

#include <cstddef>

#include "runtime/backends/cuda_abi.h"
#include "runtime/backends/hip_abi.h"
#include "runtime/backends/metal_abi.h"
#include "runtime/backends/opencl_abi.h"

namespace test {

void RunAbiTests(TestContext* ctx) {
  using lattice::runtime::cuda::ElemwiseParams;
  using lattice::runtime::cuda::FftParams;
  using lattice::runtime::cuda::LuParams;
  using lattice::runtime::cuda::MatmulParams;
  using lattice::runtime::cuda::Pool2dParams;
  using lattice::runtime::cuda::QuantileParams;
  using lattice::runtime::cuda::QrParams;
  using lattice::runtime::cuda::ReduceParams;
  using lattice::runtime::cuda::RegressionParams;
  using lattice::runtime::cuda::SolveParams;
  using lattice::runtime::cuda::SvdParams;
  using lattice::runtime::cuda::TransposeParams;
  using lattice::runtime::cuda::Conv2dParams;
  using lattice::runtime::cuda::CorrelationParams;

  ExpectTrue(lattice::runtime::cuda::IsAbiCompatible(lattice::runtime::cuda::kAbiVersion),
             "cuda_abi_compat", ctx);
  ExpectTrue(!lattice::runtime::cuda::IsAbiCompatible(
                 (lattice::runtime::cuda::kAbiVersionMajor + 1) << 16),
             "cuda_abi_wrong_major", ctx);
  ExpectTrue(lattice::runtime::hip::IsAbiCompatible(lattice::runtime::hip::kAbiVersion),
             "hip_abi_compat", ctx);
  ExpectTrue(!lattice::runtime::hip::IsAbiCompatible(
                 (lattice::runtime::hip::kAbiVersionMajor + 1) << 16),
             "hip_abi_wrong_major", ctx);
  ExpectTrue(lattice::runtime::metal::IsAbiCompatible(lattice::runtime::metal::kAbiVersion),
             "metal_abi_compat", ctx);
  ExpectTrue(!lattice::runtime::metal::IsAbiCompatible(
                 (lattice::runtime::metal::kAbiVersionMajor + 1) << 16),
             "metal_abi_wrong_major", ctx);
  ExpectTrue(lattice::runtime::opencl::IsAbiCompatible(lattice::runtime::opencl::kAbiVersion),
             "opencl_abi_compat", ctx);
  ExpectTrue(!lattice::runtime::opencl::IsAbiCompatible(
                 (lattice::runtime::opencl::kAbiVersionMajor + 1) << 16),
             "opencl_abi_wrong_major", ctx);

  ExpectTrue(sizeof(ElemwiseParams) == 280, "elemwise_params_size", ctx);
  ExpectTrue(offsetof(ElemwiseParams, count) == 0, "elemwise_params_count_off", ctx);
  ExpectTrue(offsetof(ElemwiseParams, op) == 8, "elemwise_params_op_off", ctx);
  ExpectTrue(offsetof(ElemwiseParams, dtype) == 12, "elemwise_params_dtype_off", ctx);
  ExpectTrue(offsetof(ElemwiseParams, ndim) == 16, "elemwise_params_ndim_off", ctx);
  ExpectTrue(offsetof(ElemwiseParams, flags) == 20, "elemwise_params_flags_off", ctx);
  ExpectTrue(offsetof(ElemwiseParams, shape) == 24, "elemwise_params_shape_off", ctx);

  ExpectTrue(sizeof(ReduceParams) == 24, "reduce_params_size", ctx);
  ExpectTrue(offsetof(ReduceParams, count) == 0, "reduce_params_count_off", ctx);
  ExpectTrue(offsetof(ReduceParams, op) == 8, "reduce_params_op_off", ctx);
  ExpectTrue(offsetof(ReduceParams, dtype) == 12, "reduce_params_dtype_off", ctx);
  ExpectTrue(offsetof(ReduceParams, stride) == 16, "reduce_params_stride_off", ctx);

  ExpectTrue(sizeof(MatmulParams) == 56, "matmul_params_size", ctx);
  ExpectTrue(offsetof(MatmulParams, m) == 0, "matmul_params_m_off", ctx);
  ExpectTrue(offsetof(MatmulParams, n) == 8, "matmul_params_n_off", ctx);
  ExpectTrue(offsetof(MatmulParams, k) == 16, "matmul_params_k_off", ctx);
  ExpectTrue(offsetof(MatmulParams, lda) == 24, "matmul_params_lda_off", ctx);
  ExpectTrue(offsetof(MatmulParams, ldb) == 32, "matmul_params_ldb_off", ctx);
  ExpectTrue(offsetof(MatmulParams, ldc) == 40, "matmul_params_ldc_off", ctx);
  ExpectTrue(offsetof(MatmulParams, dtype) == 48, "matmul_params_dtype_off", ctx);
  ExpectTrue(offsetof(MatmulParams, flags) == 52, "matmul_params_flags_off", ctx);

  ExpectTrue(sizeof(TransposeParams) == 16, "transpose_params_size", ctx);
  ExpectTrue(sizeof(Conv2dParams) == 48, "conv2d_params_size", ctx);
  ExpectTrue(sizeof(Pool2dParams) == 48, "pool2d_params_size", ctx);
  ExpectTrue(sizeof(FftParams) == 8, "fft_params_size", ctx);
  ExpectTrue(sizeof(SolveParams) == 16, "solve_params_size", ctx);
  ExpectTrue(sizeof(LuParams) == 8, "lu_params_size", ctx);
  ExpectTrue(sizeof(QrParams) == 16, "qr_params_size", ctx);
  ExpectTrue(sizeof(SvdParams) == 16, "svd_params_size", ctx);
  ExpectTrue(sizeof(QuantileParams) == 16, "quantile_params_size", ctx);
  ExpectTrue(sizeof(CorrelationParams) == 8, "correlation_params_size", ctx);
  ExpectTrue(sizeof(RegressionParams) == 8, "regression_params_size", ctx);

  ExpectTrue(sizeof(lattice::runtime::opencl::ElemwiseParams) == sizeof(ElemwiseParams),
             "opencl_elemwise_size_match", ctx);
  ExpectTrue(sizeof(lattice::runtime::hip::ElemwiseParams) == sizeof(ElemwiseParams),
             "hip_elemwise_size_match", ctx);
  ExpectTrue(sizeof(lattice::runtime::metal::ElemwiseParams) == sizeof(ElemwiseParams),
             "metal_elemwise_size_match", ctx);
}

}  // namespace test
