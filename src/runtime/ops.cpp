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
  if (const auto* boolean = dynamic_cast<const parser::BoolLiteral*>(&expr)) {
    return EvaluateBool(*boolean);
  }
  if (const auto* boolean = dynamic_cast<const parser::BoolLiteral*>(&expr)) {
    return EvaluateBool(*boolean);
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

ExecResult Evaluator::EvaluateStatement(parser::Statement& stmt) {
  if (auto* block = dynamic_cast<parser::BlockStatement*>(&stmt)) {
    return EvaluateBlock(*block);
  }
  if (auto* if_stmt = dynamic_cast<parser::IfStatement*>(&stmt)) {
    return EvaluateIf(*if_stmt);
  }
  if (auto* while_stmt = dynamic_cast<parser::WhileStatement*>(&stmt)) {
    return EvaluateWhile(*while_stmt);
  }
  if (auto* for_stmt = dynamic_cast<parser::ForStatement*>(&stmt)) {
    return EvaluateFor(*for_stmt);
  }
  if (dynamic_cast<parser::BreakStatement*>(&stmt) != nullptr) {
    return ExecResult{std::nullopt, ControlSignal::kBreak};
  }
  if (dynamic_cast<parser::ContinueStatement*>(&stmt) != nullptr) {
    return ExecResult{std::nullopt, ControlSignal::kContinue};
  }
  if (auto* ret = dynamic_cast<parser::ReturnStatement*>(&stmt)) {
    return EvaluateReturn(*ret);
  }
  if (auto* func = dynamic_cast<parser::FunctionStatement*>(&stmt)) {
    return EvaluateFunction(*func);
  }
  if (auto* assign = dynamic_cast<parser::AssignmentStatement*>(&stmt)) {
    return EvaluateAssignment(*assign);
  }
  if (auto* expr_stmt = dynamic_cast<parser::ExpressionStatement*>(&stmt)) {
    return EvaluateExpressionStatement(*expr_stmt);
  }
  throw std::runtime_error("Unknown statement type");
}

Value Evaluator::EvaluateNumber(const parser::NumberLiteral& literal) {
  (void)env_;
  return Value::Number(literal.value);
}

Value Evaluator::EvaluateBool(const parser::BoolLiteral& literal) {
  (void)env_;
  return Value::Bool(literal.value);
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
    case parser::BinaryOp::kEq:
      return Value::Bool(lhs.number == rhs.number);
    case parser::BinaryOp::kNe:
      return Value::Bool(lhs.number != rhs.number);
    case parser::BinaryOp::kGt:
      return Value::Bool(lhs.number > rhs.number);
    case parser::BinaryOp::kGe:
      return Value::Bool(lhs.number >= rhs.number);
    case parser::BinaryOp::kLt:
      return Value::Bool(lhs.number < rhs.number);
    case parser::BinaryOp::kLe:
      return Value::Bool(lhs.number <= rhs.number);
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
  auto found = env_->Get(name);
  if (found.has_value() && found->type == DType::kFunction) {
    auto fn = found->function;
    if (fn == nullptr) {
      throw util::Error("Function is null: " + name, 0, 0);
    }
    if (fn->body == nullptr) {
      throw util::Error("Function has no body: " + name, 0, 0);
    }
    if (fn->parameters.size() != args.size()) {
      throw util::Error(name + " expects " + std::to_string(fn->parameters.size()) +
                            " arguments",
                        0, 0);
    }
    Environment fn_env(fn->defining_env);
    for (size_t i = 0; i < args.size(); ++i) {
      fn_env.Define(fn->parameters[i], args[i]);
    }
    Evaluator fn_evaluator(&fn_env);
    ExecResult body_result = fn_evaluator.EvaluateStatement(*fn->body);
    if (body_result.control == ControlSignal::kReturn && body_result.value.has_value()) {
      return body_result.value.value();
    }
    if (body_result.value.has_value()) {
      return body_result.value.value();
    }
    return Value::Number(0.0);
  }
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

ExecResult Evaluator::EvaluateBlock(const parser::BlockStatement& block) {
  ExecResult last;
  for (const auto& stmt : block.statements) {
    ExecResult result = EvaluateStatement(*stmt);
    if (result.control != ControlSignal::kNone) {
      return result;
    }
    last = result;
  }
  return last;
}

ExecResult Evaluator::EvaluateIf(const parser::IfStatement& stmt) {
  Value condition = Evaluate(*stmt.condition);
  bool truthy = condition.number != 0.0;
  if (truthy) {
    return EvaluateStatement(*stmt.then_branch);
  }
  if (stmt.else_branch) {
    return EvaluateStatement(*stmt.else_branch);
  }
  return ExecResult{};
}

ExecResult Evaluator::EvaluateWhile(const parser::WhileStatement& stmt) {
  ExecResult last;
  while (true) {
    Value condition = Evaluate(*stmt.condition);
    if (condition.number == 0.0) {
      break;
    }
    ExecResult body = EvaluateStatement(*stmt.body);
    if (body.control == ControlSignal::kBreak) {
      body.control = ControlSignal::kNone;
      return body;
    }
    if (body.control == ControlSignal::kReturn) {
      return body;
    }
    if (body.control == ControlSignal::kContinue) {
      continue;
    }
    last = body;
  }
  return last;
}

ExecResult Evaluator::EvaluateFor(const parser::ForStatement& stmt) {
  ExecResult last;
  if (stmt.init) {
    ExecResult init_result = EvaluateStatement(*stmt.init);
    if (init_result.control != ControlSignal::kNone) {
      return init_result;
    }
  }
  while (true) {
    if (stmt.condition) {
      Value cond_val = Evaluate(*stmt.condition);
      if (cond_val.number == 0.0) {
        break;
      }
    }
    ExecResult body = EvaluateStatement(*stmt.body);
    if (body.control == ControlSignal::kBreak) {
      body.control = ControlSignal::kNone;
      return body;
    }
    if (body.control == ControlSignal::kReturn) {
      return body;
    }
    if (body.control == ControlSignal::kContinue) {
      // fall through to increment
    } else {
      last = body;
    }
    if (stmt.increment) {
      ExecResult inc = EvaluateStatement(*stmt.increment);
      if (inc.control == ControlSignal::kBreak) {
        inc.control = ControlSignal::kNone;
        return inc;
      }
      if (inc.control == ControlSignal::kReturn) {
        return inc;
      }
      if (inc.control == ControlSignal::kContinue) {
        // continue outer loop
      }
      if (inc.value.has_value()) {
        last.value = inc.value;
      }
    }
  }
  return last;
}

ExecResult Evaluator::EvaluateReturn(const parser::ReturnStatement& stmt) {
  if (stmt.expr) {
    Value v = Evaluate(*stmt.expr);
    return ExecResult{v, ControlSignal::kReturn};
  }
  return ExecResult{std::nullopt, ControlSignal::kReturn};
}

ExecResult Evaluator::EvaluateFunction(parser::FunctionStatement& stmt) {
  if (env_ == nullptr) {
    throw util::Error("Environment is not configured", 0, 0);
  }
  auto fn = std::make_shared<Function>();
  fn->parameters = stmt.parameters;
  fn->body = std::move(stmt.body);
  fn->defining_env = env_;
  env_->Define(stmt.name, Value::Func(fn));
  return ExecResult{};
}

ExecResult Evaluator::EvaluateAssignment(const parser::AssignmentStatement& stmt) {
  if (env_ == nullptr) {
    throw util::Error("Environment is not configured", 0, 0);
  }
  Value value = Evaluate(*stmt.value);
  env_->Define(stmt.name, value);
  return ExecResult{value, ControlSignal::kNone};
}

ExecResult Evaluator::EvaluateExpressionStatement(const parser::ExpressionStatement& stmt) {
  return ExecResult{Evaluate(*stmt.expr), ControlSignal::kNone};
}

std::string Value::ToString() const {
  switch (type) {
    case DType::kNumber:
      return std::to_string(number);
    case DType::kBool:
      return boolean ? "true" : "false";
    case DType::kFunction:
      return "<function>";
  }
  return "<unknown>";
}

}  // namespace lattice::runtime
