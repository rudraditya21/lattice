#include "runtime/ops.h"

#include <algorithm>
#include <cmath>
#include <numeric>
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
  if (const auto* call = dynamic_cast<const parser::CallExpression*>(&expr)) {
    return EvaluateCall(*call);
  }
  throw std::runtime_error("Unknown expression type");
}

std::optional<Value> Evaluator::EvaluateStatement(const parser::Statement& stmt) {
  if (const auto* block = dynamic_cast<const parser::BlockStatement*>(&stmt)) {
    return EvaluateBlock(*block);
  }
  if (const auto* if_stmt = dynamic_cast<const parser::IfStatement*>(&stmt)) {
    return EvaluateIf(*if_stmt);
  }
  if (const auto* assign = dynamic_cast<const parser::AssignmentStatement*>(&stmt)) {
    return EvaluateAssignment(*assign);
  }
  if (const auto* expr_stmt = dynamic_cast<const parser::ExpressionStatement*>(&stmt)) {
    return EvaluateExpressionStatement(*expr_stmt);
  }
  throw std::runtime_error("Unknown statement type");
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

Value Evaluator::EvaluateCall(const parser::CallExpression& call) {
  if (env_ == nullptr) {
    throw util::Error("Environment is not configured", 0, 0);
  }
  std::vector<Value> args;
  args.reserve(call.args.size());
  for (const auto& arg : call.args) {
    args.push_back(Evaluate(*arg));
  }
  const std::string& name = call.callee;
  auto expect_args = [&](size_t count, const std::string& func) {
    if (args.size() != count) {
      throw util::Error(func + " expects " + std::to_string(count) + " arguments", 0, 0);
    }
  };

  if (name == "pow") {
    expect_args(2, name);
    return Value::Number(std::pow(args[0].number, args[1].number));
  }
  if (name == "gcd") {
    expect_args(2, name);
    auto a = static_cast<long long>(args[0].number);
    auto b = static_cast<long long>(args[1].number);
    return Value::Number(std::gcd(a, b));
  }
  if (name == "lcm") {
    expect_args(2, name);
    auto a = static_cast<long long>(args[0].number);
    auto b = static_cast<long long>(args[1].number);
    return Value::Number(std::lcm(a, b));
  }
  if (name == "abs") {
    expect_args(1, name);
    return Value::Number(std::fabs(args[0].number));
  }
  if (name == "sign") {
    expect_args(1, name);
    double v = args[0].number;
    if (v > 0) return Value::Number(1.0);
    if (v < 0) return Value::Number(-1.0);
    return Value::Number(0.0);
  }
  if (name == "mod") {
    expect_args(2, name);
    if (args[1].number == 0.0) {
      throw util::Error("mod divisor cannot be zero", 0, 0);
    }
    return Value::Number(std::fmod(args[0].number, args[1].number));
  }
  if (name == "floor") {
    expect_args(1, name);
    return Value::Number(std::floor(args[0].number));
  }
  if (name == "ceil") {
    expect_args(1, name);
    return Value::Number(std::ceil(args[0].number));
  }
  if (name == "round") {
    expect_args(1, name);
    return Value::Number(std::round(args[0].number));
  }
  if (name == "clamp") {
    expect_args(3, name);
    return Value::Number(std::clamp(args[0].number, args[1].number, args[2].number));
  }
  if (name == "min") {
    expect_args(2, name);
    return Value::Number(std::min(args[0].number, args[1].number));
  }
  if (name == "max") {
    expect_args(2, name);
    return Value::Number(std::max(args[0].number, args[1].number));
  }
  throw util::Error("Unknown function: " + name, 0, 0);
}

std::optional<Value> Evaluator::EvaluateBlock(const parser::BlockStatement& block) {
  std::optional<Value> last;
  for (const auto& stmt : block.statements) {
    last = EvaluateStatement(*stmt);
  }
  return last;
}

std::optional<Value> Evaluator::EvaluateIf(const parser::IfStatement& stmt) {
  Value condition = Evaluate(*stmt.condition);
  bool truthy = condition.number != 0.0;
  if (truthy) {
    return EvaluateStatement(*stmt.then_branch);
  }
  if (stmt.else_branch) {
    return EvaluateStatement(*stmt.else_branch);
  }
  return std::nullopt;
}

std::optional<Value> Evaluator::EvaluateAssignment(const parser::AssignmentStatement& stmt) {
  if (env_ == nullptr) {
    throw util::Error("Environment is not configured", 0, 0);
  }
  Value value = Evaluate(*stmt.value);
  env_->Define(stmt.name, value);
  return value;
}

std::optional<Value> Evaluator::EvaluateExpressionStatement(
    const parser::ExpressionStatement& stmt) {
  return Evaluate(*stmt.expr);
}

std::string Value::ToString() const {
  switch (type) {
    case DType::kNumber:
      return std::to_string(number);
  }
  return "<unknown>";
}

}  // namespace lattice::runtime
