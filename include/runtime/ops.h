#ifndef LATTICE_RUNTIME_OPS_H_
#define LATTICE_RUNTIME_OPS_H_

#include <memory>
#include <optional>

#include "parser/ast.h"
#include "runtime/environment.h"
#include "runtime/value.h"

namespace lattice::runtime {

enum class ControlSignal { kNone, kBreak, kContinue, kReturn };

struct ExecResult {
  std::optional<Value> value;
  ControlSignal control = ControlSignal::kNone;
};

class Evaluator {
 public:
  /// Evaluates AST nodes against the provided environment (not owned).
  explicit Evaluator(Environment* env);

  /// Dispatches to the appropriate visitor for the expression kind.
  Value Evaluate(const parser::Expression& expr);

  /// Executes a statement; returns the value of the statement if any (e.g., expression statements).
  ExecResult EvaluateStatement(parser::Statement& stmt);

 private:
  Value EvaluateNumber(const parser::NumberLiteral& literal);
  Value EvaluateBool(const parser::BoolLiteral& literal);
  Value EvaluateUnary(const parser::UnaryExpression& expr);
  Value EvaluateBinary(const parser::BinaryExpression& expr);
  Value EvaluateIdentifier(const parser::Identifier& identifier);
  Value EvaluateCall(const parser::CallExpression& call);
  ExecResult EvaluateBlock(const parser::BlockStatement& block);
  ExecResult EvaluateIf(const parser::IfStatement& stmt);
  ExecResult EvaluateWhile(const parser::WhileStatement& stmt);
  ExecResult EvaluateFor(const parser::ForStatement& stmt);
  ExecResult EvaluateReturn(const parser::ReturnStatement& stmt);
  ExecResult EvaluateFunction(parser::FunctionStatement& stmt);
  ExecResult EvaluateAssignment(const parser::AssignmentStatement& stmt);
  ExecResult EvaluateExpressionStatement(const parser::ExpressionStatement& stmt);

  Environment* env_;
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_OPS_H_
