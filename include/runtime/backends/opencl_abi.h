#ifndef LATTICE_RUNTIME_BACKENDS_OPENCL_ABI_H_
#define LATTICE_RUNTIME_BACKENDS_OPENCL_ABI_H_

#include <cstdint>

namespace lattice::runtime::opencl {

constexpr uint32_t kAbiVersion = 1;

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

}  // namespace lattice::runtime::opencl

#endif  // LATTICE_RUNTIME_BACKENDS_OPENCL_ABI_H_
