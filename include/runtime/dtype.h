#ifndef LATTICE_RUNTIME_DTYPE_H_
#define LATTICE_RUNTIME_DTYPE_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lattice::runtime {

enum class TensorKind { kDense, kSparseCSR, kSparseCOO, kRagged };

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
  kString,
  kDecimal,
  kRational,
  kFunction,
  kTensor,
  kTuple,
  kRecord
};

struct Type {
  DType kind = DType::kBool;
  // Optional element dtypes for tuples; empty means unknown length in dynamic mode.
  std::vector<std::optional<DType>> tuple_elems;
  // Ordered fields for records: name -> dtype (nullopt means dynamic).
  std::vector<std::pair<std::string, std::optional<DType>>> record_fields;
  // Optional tensor kind metadata.
  std::optional<TensorKind> tensor_kind;

  static Type Tuple(std::vector<std::optional<DType>> elems) {
    Type t;
    t.kind = DType::kTuple;
    t.tuple_elems = std::move(elems);
    return t;
  }

  static Type Record(std::vector<std::pair<std::string, std::optional<DType>>> fields) {
    Type t;
    t.kind = DType::kRecord;
    t.record_fields = std::move(fields);
    return t;
  }
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_DTYPE_H_
