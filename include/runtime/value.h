#ifndef LATTICE_RUNTIME_VALUE_H_
#define LATTICE_RUNTIME_VALUE_H_

#include <string>

#include "runtime/dtype.h"

namespace lattice::runtime {

struct Value {
  DType type;
  double number;

  /// Convenience constructor for numeric values.
  static Value Number(double v) { return Value{DType::kNumber, v}; }

  /// Formats the value for display in the REPL.
  std::string ToString() const;
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_VALUE_H_
