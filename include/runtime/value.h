#ifndef LATTICE_RUNTIME_VALUE_H_
#define LATTICE_RUNTIME_VALUE_H_

#include <memory>
#include <string>
#include <vector>

#include "parser/ast.h"
#include "runtime/dtype.h"

namespace lattice {
namespace parser {
struct Statement;
}  // namespace parser
namespace runtime {

class Environment;

struct Function {
  std::vector<std::string> parameters;
  std::unique_ptr<parser::Statement> body;
  Environment* defining_env;
};

struct Value {
  DType type;
  double number;
  bool boolean;
  std::shared_ptr<Function> function;

  /// Convenience constructor for numeric values.
  static Value Number(double v) { return Value{DType::kNumber, v, v != 0.0, nullptr}; }
  /// Convenience constructor for boolean values.
  static Value Bool(bool v) { return Value{DType::kBool, v ? 1.0 : 0.0, v, nullptr}; }
  /// Convenience constructor for function values.
  static Value Func(std::shared_ptr<Function> fn);

  /// Formats the value for display in the REPL.
  std::string ToString() const;
};

inline Value Value::Func(std::shared_ptr<Function> fn) {
  return Value{DType::kFunction, 0.0, false, std::move(fn)};
}

}  // namespace runtime
}  // namespace lattice

#endif  // LATTICE_RUNTIME_VALUE_H_
