#include "runtime/ops.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "util/error.h"

namespace lattice::runtime {

namespace {
bool IsBoolTypeName(const std::string& name) {
  return name == "bool";
}

bool IsNumericTypeName(const std::string& name) {
  static const std::unordered_set<std::string> kNumeric = {
      "i8",  "i16", "i32", "i64",      "u8",        "u16",        "u32",     "u64",
      "f16", "f32", "f64", "bfloat16", "complex64", "complex128", "decimal", "rational"};
  return kNumeric.find(name) != kNumeric.end();
}

bool ValueMatchesType(const Value& value, const std::string& type_name) {
  if (type_name.empty()) {
    return true;
  }
  if (IsBoolTypeName(type_name)) {
    return value.type == DType::kBool;
  }
  if (IsNumericTypeName(type_name)) {
    if (value.type == DType::kBool || value.type == DType::kFunction) return false;
    if (value.type_name.empty()) return true;
    if (IsNumericTypeName(value.type_name)) return true;
    return value.type_name == type_name;
  }
  // For now, unsupported or unknown types fail.
  return false;
}

std::string CombineNumericType(const Value& lhs, const Value& rhs) {
  auto is_floatish = [](const std::string& t) {
    return t == "f16" || t == "f32" || t == "f64" || t == "bfloat16" || t.rfind("complex", 0) == 0;
  };
  auto is_intish = [](const std::string& t) { return !t.empty() && (t[0] == 'i' || t[0] == 'u'); };
  const std::string& lt = lhs.type_name;
  const std::string& rt = rhs.type_name;
  if (is_floatish(lt) || is_floatish(rt)) {
    return "f64";  // default float promotion
  }
  if (is_intish(lt)) return lt;
  if (is_intish(rt)) return rt;
  return "f64";
}

std::string DeriveTypeName(const Value& v) {
  if (!v.type_name.empty()) return v.type_name;
  if (v.type == DType::kBool) return "bool";
  if (v.type == DType::kI32 || v.type == DType::kI64 || v.type == DType::kU32 ||
      v.type == DType::kU64) {
    return v.type_name;
  }
  if (v.type == DType::kF32 || v.type == DType::kF64) {
    return v.type_name;
  }
  if (v.type == DType::kDecimal || v.type == DType::kRational) {
    return v.type_name;
  }
  {
    double intpart;
    if (std::modf(v.f64, &intpart) == 0.0) {
      // Integral literal: assume i32 if in range.
      if (v.f64 >= -2147483648.0 && v.f64 <= 2147483647.0) {
        return "i32";
      }
      return "i64";
    }
    return "f64";
  }
  return "";
}
}  // namespace

Evaluator::Evaluator(Environment* env) : env_(env) {}

Value Evaluator::Evaluate(const parser::Expression& expr) {
  if (const auto* num = dynamic_cast<const parser::NumberLiteral*>(&expr)) {
    return EvaluateNumber(*num);
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
  std::string tn;
  if (literal.is_integer_token) {
    return Value::I32(static_cast<int32_t>(literal.value));
  }
  size_t digits = 0;
  for (char c : literal.lexeme) {
    if (std::isdigit(static_cast<unsigned char>(c))) {
      ++digits;
    }
  }
  if (digits <= 7) {
    return Value::F32(static_cast<float>(literal.value));
  }
  return Value::F64(literal.value);
}

Value Evaluator::EvaluateBool(const parser::BoolLiteral& literal) {
  (void)env_;
  return Value::Bool(literal.value);
}

Value Evaluator::EvaluateUnary(const parser::UnaryExpression& expr) {
  Value operand = Evaluate(*expr.operand);
  switch (expr.op) {
    case parser::UnaryOp::kNegate:
      switch (operand.type) {
        case DType::kBool:
          return Value::Bool(!operand.boolean);
        case DType::kI32:
        case DType::kI64:
          return Value::I64(-operand.i64);
        case DType::kU32:
        case DType::kU64:
          return Value::I64(-static_cast<int64_t>(operand.u64));
        case DType::kF32:
          return Value::F32(-static_cast<float>(operand.f64));
        case DType::kF64:
          return Value::F64(-operand.f64);
        default:
          return Value::F64(-operand.f64);
      }
  }
  throw std::runtime_error("Unhandled unary operator");
}

Value Evaluator::EvaluateBinary(const parser::BinaryExpression& expr) {
  Value lhs = Evaluate(*expr.lhs);
  Value rhs = Evaluate(*expr.rhs);
  auto is_int = [](DType t) {
    return t == DType::kI8 || t == DType::kI16 || t == DType::kI32 || t == DType::kI64 ||
           t == DType::kU8 || t == DType::kU16 || t == DType::kU32 || t == DType::kU64;
  };
  auto is_float = [](DType t) {
    return t == DType::kF16 || t == DType::kBF16 || t == DType::kF32 || t == DType::kF64;
  };
  auto is_complex = [](DType t) { return t == DType::kC64 || t == DType::kC128; };

  if (is_complex(lhs.type) || is_complex(rhs.type)) {
    std::complex<double> lcv =
        is_complex(lhs.type) ? lhs.complex : std::complex<double>(lhs.f64, 0.0);
    std::complex<double> rcv =
        is_complex(rhs.type) ? rhs.complex : std::complex<double>(rhs.f64, 0.0);
    switch (expr.op) {
      case parser::BinaryOp::kAdd:
        return Value::Complex128(lcv + rcv);
      case parser::BinaryOp::kSub:
        return Value::Complex128(lcv - rcv);
      case parser::BinaryOp::kMul:
        return Value::Complex128(lcv * rcv);
      case parser::BinaryOp::kDiv:
        return Value::Complex128(lcv / rcv);
      case parser::BinaryOp::kEq:
        return Value::Bool(lcv == rcv);
      case parser::BinaryOp::kNe:
        return Value::Bool(lcv != rcv);
      case parser::BinaryOp::kGt:
      case parser::BinaryOp::kGe:
      case parser::BinaryOp::kLt:
      case parser::BinaryOp::kLe:
        throw util::Error("Complex comparison not supported", 0, 0);
    }
  }

  if (is_float(lhs.type) || is_float(rhs.type)) {
    double lv = lhs.f64;
    double rv = rhs.f64;
    switch (expr.op) {
      case parser::BinaryOp::kAdd:
        return Value::F64(lv + rv);
      case parser::BinaryOp::kSub:
        return Value::F64(lv - rv);
      case parser::BinaryOp::kMul:
        return Value::F64(lv * rv);
      case parser::BinaryOp::kDiv:
        return Value::F64(lv / rv);
      case parser::BinaryOp::kEq:
        return Value::Bool(lv == rv);
      case parser::BinaryOp::kNe:
        return Value::Bool(lv != rv);
      case parser::BinaryOp::kGt:
        return Value::Bool(lv > rv);
      case parser::BinaryOp::kGe:
        return Value::Bool(lv >= rv);
      case parser::BinaryOp::kLt:
        return Value::Bool(lv < rv);
      case parser::BinaryOp::kLe:
        return Value::Bool(lv <= rv);
    }
  }

  if (is_int(lhs.type) && is_int(rhs.type)) {
    int64_t lv = lhs.i64;
    int64_t rv = rhs.i64;
    std::string res_type =
        (lhs.type_name == rhs.type_name && !lhs.type_name.empty()) ? lhs.type_name : "i64";
    switch (expr.op) {
      case parser::BinaryOp::kAdd:
        if (res_type == "i32") {
          auto v = Value::I32(static_cast<int32_t>(lv + rv));
          v.type_name = res_type;
          return v;
        }
        return Value::I64(lv + rv);
      case parser::BinaryOp::kSub:
        if (res_type == "i32") {
          auto v = Value::I32(static_cast<int32_t>(lv - rv));
          v.type_name = res_type;
          return v;
        }
        return Value::I64(lv - rv);
      case parser::BinaryOp::kMul:
        if (res_type == "i32") {
          auto v = Value::I32(static_cast<int32_t>(lv * rv));
          v.type_name = res_type;
          return v;
        }
        return Value::I64(lv * rv);
      case parser::BinaryOp::kDiv:
        return Value::F64(static_cast<double>(lv) / static_cast<double>(rv));
      case parser::BinaryOp::kEq:
        return Value::Bool(lv == rv);
      case parser::BinaryOp::kNe:
        return Value::Bool(lv != rv);
      case parser::BinaryOp::kGt:
        return Value::Bool(lv > rv);
      case parser::BinaryOp::kGe:
        return Value::Bool(lv >= rv);
      case parser::BinaryOp::kLt:
        return Value::Bool(lv < rv);
      case parser::BinaryOp::kLe:
        return Value::Bool(lv <= rv);
    }
  }

  // Fallback to double.
  double lv = lhs.f64;
  double rv = rhs.f64;
  switch (expr.op) {
    case parser::BinaryOp::kAdd:
      return Value::F64(lv + rv);
    case parser::BinaryOp::kSub:
      return Value::F64(lv - rv);
    case parser::BinaryOp::kMul:
      return Value::F64(lv * rv);
    case parser::BinaryOp::kDiv:
      return Value::F64(lv / rv);
    case parser::BinaryOp::kEq:
      return Value::Bool(lv == rv);
    case parser::BinaryOp::kNe:
      return Value::Bool(lv != rv);
    case parser::BinaryOp::kGt:
      return Value::Bool(lv > rv);
    case parser::BinaryOp::kGe:
      return Value::Bool(lv >= rv);
    case parser::BinaryOp::kLt:
      return Value::Bool(lv < rv);
    case parser::BinaryOp::kLe:
      return Value::Bool(lv <= rv);
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
      if (name == "print") {
        if (args.size() != 1) {
          throw util::Error("print expects 1 argument", 0, 0);
        }
        std::cout << args[0].ToString() << "\n";
        return Value::Number(0.0);
      }
      throw util::Error("Function has no body: " + name, 0, 0);
    }
    if (fn->parameters.size() != args.size()) {
      throw util::Error(name + " expects " + std::to_string(fn->parameters.size()) + " arguments",
                        0, 0);
    }
    Environment fn_env(fn->defining_env);
    for (size_t i = 0; i < args.size(); ++i) {
      if (!fn->parameter_types.empty() && i < fn->parameter_types.size()) {
        const std::string& annot = fn->parameter_types[i];
        if (!annot.empty() && !ValueMatchesType(args[i], annot)) {
          throw util::Error("Type mismatch for parameter '" + fn->parameters[i] + "'", 0, 0);
        }
      }
      fn_env.Define(fn->parameters[i], args[i]);
    }
    Evaluator fn_evaluator(&fn_env);
    ExecResult body_result = fn_evaluator.EvaluateStatement(*fn->body);
    if (!fn->return_type.empty()) {
      if (body_result.control == ControlSignal::kReturn && body_result.value.has_value()) {
        if (!ValueMatchesType(body_result.value.value(), fn->return_type)) {
          throw util::Error("Return type mismatch in function " + name, 0, 0);
        }
        body_result.value->type_name = fn->return_type;
      } else if (body_result.value.has_value()) {
        if (!ValueMatchesType(body_result.value.value(), fn->return_type)) {
          throw util::Error("Return type mismatch in function " + name, 0, 0);
        }
        body_result.value->type_name = fn->return_type;
      }
    }
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
    return Value::F64(std::pow(args[0].f64, args[1].f64));
  }
  if (name == "gcd") {
    expect_args(2, name);
    auto a = static_cast<long long>(args[0].f64);
    auto b = static_cast<long long>(args[1].f64);
    return Value::I64(std::gcd(a, b));
  }
  if (name == "lcm") {
    expect_args(2, name);
    auto a = static_cast<long long>(args[0].f64);
    auto b = static_cast<long long>(args[1].f64);
    return Value::I64(std::lcm(a, b));
  }
  if (name == "abs") {
    expect_args(1, name);
    return Value::F64(std::fabs(args[0].f64));
  }
  if (name == "sign") {
    expect_args(1, name);
    double v = args[0].f64;
    if (v > 0) return Value::I32(1);
    if (v < 0) return Value::I32(-1);
    return Value::I32(0);
  }
  if (name == "mod") {
    expect_args(2, name);
    if (args[1].f64 == 0.0) {
      throw util::Error("mod divisor cannot be zero", 0, 0);
    }
    return Value::F64(std::fmod(args[0].f64, args[1].f64));
  }
  if (name == "floor") {
    expect_args(1, name);
    return Value::F64(std::floor(args[0].f64));
  }
  if (name == "ceil") {
    expect_args(1, name);
    return Value::F64(std::ceil(args[0].f64));
  }
  if (name == "round") {
    expect_args(1, name);
    return Value::F64(std::round(args[0].f64));
  }
  if (name == "clamp") {
    expect_args(3, name);
    return Value::F64(std::clamp(args[0].f64, args[1].f64, args[2].f64));
  }
  if (name == "min") {
    expect_args(2, name);
    return Value::F64(std::min(args[0].f64, args[1].f64));
  }
  if (name == "max") {
    expect_args(2, name);
    return Value::F64(std::max(args[0].f64, args[1].f64));
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
  bool truthy = condition.boolean || condition.f64 != 0.0;
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
    if (condition.f64 == 0.0) {
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
      if (cond_val.f64 == 0.0) {
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
  fn->parameter_types.reserve(stmt.parameter_types.size());
  for (const auto& ann : stmt.parameter_types) {
    if (ann.type) {
      fn->parameter_types.push_back(ann.type->name);
    } else {
      fn->parameter_types.push_back("");
    }
  }
  if (stmt.return_type.type) {
    fn->return_type = stmt.return_type.type->name;
  }
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
  if (stmt.annotation.type) {
    if (!ValueMatchesType(value, stmt.annotation.type->name)) {
      throw util::Error("Type mismatch for variable '" + stmt.name + "'", 0, 0);
    }
    value.type_name = stmt.annotation.type->name;
  }
  env_->Define(stmt.name, value);
  return ExecResult{value, ControlSignal::kNone};
}

ExecResult Evaluator::EvaluateExpressionStatement(const parser::ExpressionStatement& stmt) {
  return ExecResult{Evaluate(*stmt.expr), ControlSignal::kNone};
}

std::string Value::ToString() const {
  switch (type) {
    case DType::kBool:
      return boolean ? "true" : "false";
    case DType::kI8:
    case DType::kI16:
    case DType::kI32:
    case DType::kI64:
      return std::to_string(i64);
    case DType::kU8:
    case DType::kU16:
    case DType::kU32:
    case DType::kU64:
      return std::to_string(u64);
    case DType::kF16:
    case DType::kBF16:
    case DType::kF32:
    case DType::kF64:
      return std::to_string(f64);
    case DType::kC64:
    case DType::kC128: {
      std::ostringstream oss;
      oss << complex.real() << "+" << complex.imag() << "i";
      return oss.str();
    }
    case DType::kDecimal:
      return std::to_string(static_cast<double>(decimal));
    case DType::kRational: {
      std::ostringstream oss;
      oss << rational.num << "/" << rational.den;
      return oss.str();
    }
    case DType::kFunction:
      return "<function>";
  }
  return "<unknown>";
}

}  // namespace lattice::runtime
