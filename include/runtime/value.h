#ifndef LATTICE_RUNTIME_VALUE_H_
#define LATTICE_RUNTIME_VALUE_H_

#include <string>

#include "runtime/dtype.h"

namespace lattice::runtime {

struct Value {
  DType type;
  double number;

  static Value Number(double v) { return Value{DType::kNumber, v}; }

  std::string ToString() const;
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_VALUE_H_
