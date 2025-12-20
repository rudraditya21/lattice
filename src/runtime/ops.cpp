#include "runtime/ops.h"

#include <stdexcept>

#include "util/error.h"

namespace lattice::runtime {

Evaluator::Evaluator(Environment* env) : env_(env) {}

Value Evaluator::Evaluate(const parser::Expression& expr) {
  if (const auto* num = dynamic_cast<const parser::NumberLiteral*>(&expr)) {
    return EvaluateNumber(*num);
  }
  if (const auto* unary = dynamic_cast<const parser::UnaryExpression*>(&expr)) {
    return EvaluateUnary(*unary);
  }
  if (const auto* binary = dynamic_cast<const parser::BinaryExpression*>(&expr)) {
    return EvaluateBinary(*binary);
  }
  if (const auto* identifier = dynamic_cast<const parser::Identifier*>(&expr)) {
    return EvaluateIdentifier(*identifier);
  }
  throw std::runtime_error("Unknown expression type");
}

Value Evaluator::EvaluateNumber(const parser::NumberLiteral& literal) {
  (void)env_;
  return Value::Number(literal.value);
}

Value Evaluator::EvaluateUnary(const parser::UnaryExpression& expr) {
  Value operand = Evaluate(*expr.operand);
  switch (expr.op) {
    case parser::UnaryOp::kNegate:
      return Value::Number(-operand.number);
  }
  throw std::runtime_error("Unhandled unary operator");
}

Value Evaluator::EvaluateBinary(const parser::BinaryExpression& expr) {
  Value lhs = Evaluate(*expr.lhs);
  Value rhs = Evaluate(*expr.rhs);
  switch (expr.op) {
    case parser::BinaryOp::kAdd:
      return Value::Number(lhs.number + rhs.number);
    case parser::BinaryOp::kSub:
      return Value::Number(lhs.number - rhs.number);
    case parser::BinaryOp::kMul:
      return Value::Number(lhs.number * rhs.number);
    case parser::BinaryOp::kDiv:
      return Value::Number(lhs.number / rhs.number);
  }
  throw std::runtime_error("Unhandled binary operator");
}

Value Evaluator::EvaluateIdentifier(const parser::Identifier& identifier) {
  if (env_ == nullptr) {
    throw util::Error("Environment is not configured", 0, 0);
  }
  auto value = env_->Get(identifier.name);
  if (!value.has_value()) {
    throw util::Error("Undefined identifier: " + identifier.name, 0, 0);
  }
  return value.value();
}

std::string Value::ToString() const {
  switch (type) {
    case DType::kNumber:
      return std::to_string(number);
  }
  return "<unknown>";
}

}  // namespace lattice::runtime
