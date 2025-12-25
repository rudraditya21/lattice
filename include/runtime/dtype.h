#ifndef LATTICE_RUNTIME_DTYPE_H_
#define LATTICE_RUNTIME_DTYPE_H_

namespace lattice::runtime {

enum class DType {
  kBool,
  kI8,
  kI16,
  kI32,
  kI64,
  kU8,
  kU16,
  kU32,
  kU64,
  kF16,
  kBF16,
  kF32,
  kF64,
  kC64,
  kC128,
  kDecimal,
  kRational,
  kFunction
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_DTYPE_H_
