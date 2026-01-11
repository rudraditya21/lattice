#ifndef LATTICE_RUNTIME_BACKENDS_CUDA_ABI_H_
#define LATTICE_RUNTIME_BACKENDS_CUDA_ABI_H_

#include <cstddef>
#include <cstdint>

namespace lattice::runtime::cuda {

constexpr uint32_t kAbiVersionMajor = 1;
constexpr uint32_t kAbiVersionMinor = 1;
constexpr uint32_t kAbiVersion = (kAbiVersionMajor << 16) | kAbiVersionMinor;
constexpr uint32_t kAbiVersionMinMajor = 1;
constexpr uint32_t kAbiVersionMinMinor = 1;
constexpr uint32_t kAbiVersionMin = (kAbiVersionMinMajor << 16) | kAbiVersionMinMinor;

constexpr bool IsAbiCompatible(uint32_t version) {
  return ((version >> 16) == kAbiVersionMajor) && version >= kAbiVersionMin;
}

enum class ElemwiseOp : uint32_t {
  kAdd = 1,
  kSub = 2,
  kMul = 3,
  kDiv = 4,
};

enum class ReduceOp : uint32_t {
  kSum = 1,
  kMin = 2,
  kMax = 3,
};

enum class DTypeCode : uint32_t {
  kF32 = 1,
  kF64 = 2,
  kI32 = 3,
  kU32 = 4,
};

constexpr uint32_t kMaxTensorDims = 8;

// Fixed ABI: kernels receive input buffers first, then a params struct by value.
struct ElemwiseParams {
  uint64_t count = 0;
  uint32_t op = 0;
  uint32_t dtype = 0;
  uint32_t ndim = 0;
  uint32_t flags = 0;
  uint64_t shape[kMaxTensorDims] = {};
  uint64_t out_strides[kMaxTensorDims] = {};
  uint64_t lhs_strides[kMaxTensorDims] = {};
  uint64_t rhs_strides[kMaxTensorDims] = {};
};

struct ReduceParams {
  uint64_t count = 0;
  uint32_t op = 0;
  uint32_t dtype = 0;
  uint64_t stride = 0;
};

struct MatmulParams {
  uint64_t m = 0;
  uint64_t n = 0;
  uint64_t k = 0;
  uint64_t lda = 0;
  uint64_t ldb = 0;
  uint64_t ldc = 0;
  uint32_t dtype = 0;
  uint32_t flags = 0;
};

struct TransposeParams {
  uint64_t rows = 0;
  uint64_t cols = 0;
};

struct Conv2dParams {
  uint64_t in_h = 0;
  uint64_t in_w = 0;
  uint64_t k_h = 0;
  uint64_t k_w = 0;
  uint64_t out_h = 0;
  uint64_t out_w = 0;
};

struct Pool2dParams {
  uint64_t in_h = 0;
  uint64_t in_w = 0;
  uint64_t k_h = 0;
  uint64_t k_w = 0;
  uint64_t out_h = 0;
  uint64_t out_w = 0;
};

struct FftParams {
  uint64_t n = 0;
};

struct SolveParams {
  uint64_t n = 0;
  uint64_t rhs_cols = 0;
};

struct LuParams {
  uint64_t n = 0;
};

struct QrParams {
  uint64_t m = 0;
  uint64_t n = 0;
};

struct SvdParams {
  uint64_t m = 0;
  uint64_t n = 0;
};

struct QuantileParams {
  uint64_t count = 0;
  float q = 0.0f;
  uint32_t pad = 0;
};

struct CorrelationParams {
  uint64_t count = 0;
};

struct RegressionParams {
  uint64_t count = 0;
};

static_assert(sizeof(ElemwiseParams) == 280, "ElemwiseParams size mismatch");
static_assert(offsetof(ElemwiseParams, count) == 0, "ElemwiseParams.count offset mismatch");
static_assert(offsetof(ElemwiseParams, op) == 8, "ElemwiseParams.op offset mismatch");
static_assert(offsetof(ElemwiseParams, dtype) == 12, "ElemwiseParams.dtype offset mismatch");
static_assert(offsetof(ElemwiseParams, ndim) == 16, "ElemwiseParams.ndim offset mismatch");
static_assert(offsetof(ElemwiseParams, flags) == 20, "ElemwiseParams.flags offset mismatch");
static_assert(offsetof(ElemwiseParams, shape) == 24, "ElemwiseParams.shape offset mismatch");
static_assert(offsetof(ElemwiseParams, out_strides) == 24 + 8 * kMaxTensorDims,
              "ElemwiseParams.out_strides offset mismatch");
static_assert(offsetof(ElemwiseParams, lhs_strides) == 24 + 16 * kMaxTensorDims,
              "ElemwiseParams.lhs_strides offset mismatch");
static_assert(offsetof(ElemwiseParams, rhs_strides) == 24 + 24 * kMaxTensorDims,
              "ElemwiseParams.rhs_strides offset mismatch");

static_assert(sizeof(ReduceParams) == 24, "ReduceParams size mismatch");
static_assert(offsetof(ReduceParams, count) == 0, "ReduceParams.count offset mismatch");
static_assert(offsetof(ReduceParams, op) == 8, "ReduceParams.op offset mismatch");
static_assert(offsetof(ReduceParams, dtype) == 12, "ReduceParams.dtype offset mismatch");
static_assert(offsetof(ReduceParams, stride) == 16, "ReduceParams.stride offset mismatch");

static_assert(sizeof(MatmulParams) == 56, "MatmulParams size mismatch");
static_assert(offsetof(MatmulParams, m) == 0, "MatmulParams.m offset mismatch");
static_assert(offsetof(MatmulParams, n) == 8, "MatmulParams.n offset mismatch");
static_assert(offsetof(MatmulParams, k) == 16, "MatmulParams.k offset mismatch");
static_assert(offsetof(MatmulParams, lda) == 24, "MatmulParams.lda offset mismatch");
static_assert(offsetof(MatmulParams, ldb) == 32, "MatmulParams.ldb offset mismatch");
static_assert(offsetof(MatmulParams, ldc) == 40, "MatmulParams.ldc offset mismatch");
static_assert(offsetof(MatmulParams, dtype) == 48, "MatmulParams.dtype offset mismatch");
static_assert(offsetof(MatmulParams, flags) == 52, "MatmulParams.flags offset mismatch");

static_assert(sizeof(TransposeParams) == 16, "TransposeParams size mismatch");
static_assert(sizeof(Conv2dParams) == 48, "Conv2dParams size mismatch");
static_assert(sizeof(Pool2dParams) == 48, "Pool2dParams size mismatch");
static_assert(sizeof(FftParams) == 8, "FftParams size mismatch");
static_assert(sizeof(SolveParams) == 16, "SolveParams size mismatch");
static_assert(sizeof(LuParams) == 8, "LuParams size mismatch");
static_assert(sizeof(QrParams) == 16, "QrParams size mismatch");
static_assert(sizeof(SvdParams) == 16, "SvdParams size mismatch");
static_assert(sizeof(QuantileParams) == 16, "QuantileParams size mismatch");
static_assert(sizeof(CorrelationParams) == 8, "CorrelationParams size mismatch");
static_assert(sizeof(RegressionParams) == 8, "RegressionParams size mismatch");

}  // namespace lattice::runtime::cuda

#endif  // LATTICE_RUNTIME_BACKENDS_CUDA_ABI_H_
