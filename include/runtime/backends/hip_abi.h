#ifndef LATTICE_RUNTIME_BACKENDS_HIP_ABI_H_
#define LATTICE_RUNTIME_BACKENDS_HIP_ABI_H_

#include <cstddef>
#include <cstdint>

namespace lattice::runtime::hip {

constexpr uint32_t kAbiVersionMajor = 1;
constexpr uint32_t kAbiVersionMinor = 0;
constexpr uint32_t kAbiVersion = (kAbiVersionMajor << 16) | kAbiVersionMinor;
constexpr uint32_t kAbiVersionMinMajor = 1;
constexpr uint32_t kAbiVersionMinMinor = 0;
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

// Fixed ABI: kernels receive input buffers first, then a params struct by value.
struct ElemwiseParams {
  uint64_t count = 0;
  uint32_t op = 0;
  uint32_t dtype = 0;
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

static_assert(sizeof(ElemwiseParams) == 16, "ElemwiseParams size mismatch");
static_assert(offsetof(ElemwiseParams, count) == 0, "ElemwiseParams.count offset mismatch");
static_assert(offsetof(ElemwiseParams, op) == 8, "ElemwiseParams.op offset mismatch");
static_assert(offsetof(ElemwiseParams, dtype) == 12, "ElemwiseParams.dtype offset mismatch");

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

}  // namespace lattice::runtime::hip

#endif  // LATTICE_RUNTIME_BACKENDS_HIP_ABI_H_
