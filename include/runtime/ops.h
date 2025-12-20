#ifndef LATTICE_RUNTIME_OPS_H_
#define LATTICE_RUNTIME_OPS_H_

#include <memory>

#include "parser/ast.h"
#include "runtime/environment.h"
#include "runtime/value.h"

namespace lattice::runtime {

class Evaluator {
 public:
  /// Evaluates AST nodes against the provided environment (not owned).
  explicit Evaluator(Environment* env);

  /// Dispatches to the appropriate visitor for the expression kind.
  Value Evaluate(const parser::Expression& expr);

 private:
  Value EvaluateNumber(const parser::NumberLiteral& literal);
  Value EvaluateUnary(const parser::UnaryExpression& expr);
  Value EvaluateBinary(const parser::BinaryExpression& expr);
  Value EvaluateIdentifier(const parser::Identifier& identifier);
  Value EvaluateCall(const parser::CallExpression& call);

  Environment* env_;
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_OPS_H_
