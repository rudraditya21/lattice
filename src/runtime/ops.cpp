#include "runtime/ops.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "runtime/decimal.h"
#include "util/error.h"

namespace lattice::runtime {

bool IsComplex(DType t) {
  return t == DType::kC64 || t == DType::kC128;
}
bool IsFloat(DType t) {
  return t == DType::kF16 || t == DType::kBF16 || t == DType::kF32 || t == DType::kF64;
}
bool IsSignedInt(DType t) {
  return t == DType::kI8 || t == DType::kI16 || t == DType::kI32 || t == DType::kI64;
}
bool IsUnsignedInt(DType t) {
  return t == DType::kU8 || t == DType::kU16 || t == DType::kU32 || t == DType::kU64;
}

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

int ComplexRank(DType t) {
  if (t == DType::kC128) return 2;
  if (t == DType::kC64) return 1;
  return 0;
}

int FloatRank(DType t) {
  if (t == DType::kF64) return 4;
  if (t == DType::kF32) return 3;
  if (t == DType::kBF16) return 2;
  if (t == DType::kF16) return 1;
  return 0;
}

std::optional<DType> LookupDType(const std::string& name) {
  if (name == "bool") return DType::kBool;
  if (name == "i8") return DType::kI8;
  if (name == "i16") return DType::kI16;
  if (name == "i32") return DType::kI32;
  if (name == "i64") return DType::kI64;
  if (name == "u8") return DType::kU8;
  if (name == "u16") return DType::kU16;
  if (name == "u32") return DType::kU32;
  if (name == "u64") return DType::kU64;
  if (name == "f16") return DType::kF16;
  if (name == "bfloat16") return DType::kBF16;
  if (name == "f32") return DType::kF32;
  if (name == "f64") return DType::kF64;
  if (name == "complex64") return DType::kC64;
  if (name == "complex128") return DType::kC128;
  if (name == "decimal") return DType::kDecimal;
  if (name == "rational") return DType::kRational;
  return std::nullopt;
}

int SignedRank(DType t) {
  if (t == DType::kI64) return 4;
  if (t == DType::kI32) return 3;
  if (t == DType::kI16) return 2;
  if (t == DType::kI8) return 1;
  return 0;
}

int UnsignedRank(DType t) {
  if (t == DType::kU64) return 4;
  if (t == DType::kU32) return 3;
  if (t == DType::kU16) return 2;
  if (t == DType::kU8) return 1;
  return 0;
}

DType RankToSigned(int rank) {
  switch (rank) {
    case 1:
      return DType::kI8;
    case 2:
      return DType::kI16;
    case 3:
      return DType::kI32;
    default:
      return DType::kI64;
  }
}

DType RankToUnsigned(int rank) {
  switch (rank) {
    case 1:
      return DType::kU8;
    case 2:
      return DType::kU16;
    case 3:
      return DType::kU32;
    default:
      return DType::kU64;
  }
}

DType RankToFloat(int rank, bool seen_bf16) {
  switch (rank) {
    case 1:
      return DType::kF16;
    case 2:
      return seen_bf16 ? DType::kBF16 : DType::kF16;
    case 3:
      return DType::kF32;
    default:
      return DType::kF64;
  }
}

DType PromoteType(DType a, DType b) {
  auto is_decimal_or_rational = [](DType t) {
    return t == DType::kDecimal || t == DType::kRational;
  };
  if (a == DType::kDecimal && b == DType::kDecimal) return DType::kDecimal;
  if (a == DType::kRational && b == DType::kRational) return DType::kRational;
  if (is_decimal_or_rational(a) || is_decimal_or_rational(b)) {
    throw util::Error("Decimal/rational cross-type arithmetic not supported in promotion", 0, 0);
  }
  const int complex_rank = std::max(ComplexRank(a), ComplexRank(b));
  const int float_rank = std::max(FloatRank(a), FloatRank(b));
  const bool seen_bf16 = (a == DType::kBF16) || (b == DType::kBF16);
  if (complex_rank > 0 || IsComplex(a) || IsComplex(b)) {
    if (complex_rank == 2 || float_rank >= 4) {
      return DType::kC128;
    }
    if (float_rank >= 3) {
      return DType::kC128;
    }
    if (float_rank >= 1 && complex_rank == 0) {
      // Complex wins; choose width based on float precision.
      return float_rank >= 2 ? DType::kC128 : DType::kC64;
    }
    return complex_rank == 1 ? DType::kC64 : DType::kC128;
  }
  if (float_rank > 0) {
    return RankToFloat(float_rank, seen_bf16);
  }
  const int signed_rank = std::max(SignedRank(a), SignedRank(b));
  const int unsigned_rank = std::max(UnsignedRank(a), UnsignedRank(b));
  if (signed_rank > 0 && unsigned_rank > 0) {
    if (unsigned_rank >= signed_rank) {
      return DType::kI64;
    }
    return RankToSigned(std::max(signed_rank, unsigned_rank));
  }
  if (signed_rank > 0) return RankToSigned(signed_rank);
  if (unsigned_rank > 0) return RankToUnsigned(unsigned_rank);
  return a;
}

Value CastTo(DType target, const Value& v) {
  auto as_double = [&]() -> double {
    switch (v.type) {
      case DType::kBool:
        return v.boolean ? 1.0 : 0.0;
      case DType::kI8:
      case DType::kI16:
      case DType::kI32:
      case DType::kI64:
        return static_cast<double>(v.i64);
      case DType::kU8:
      case DType::kU16:
      case DType::kU32:
      case DType::kU64:
        return static_cast<double>(v.u64);
      case DType::kF16:
      case DType::kBF16:
      case DType::kF32:
      case DType::kF64:
        return v.f64;
      case DType::kDecimal:
        return static_cast<double>(v.decimal);
      case DType::kRational:
        return static_cast<double>(v.rational.num) / static_cast<double>(v.rational.den);
      default:
        throw util::Error("Cannot cast complex/decimal/rational/function", 0, 0);
    }
  };
  auto as_signed = [&]() -> int64_t {
    switch (v.type) {
      case DType::kBool:
        return v.boolean ? 1 : 0;
      case DType::kI8:
      case DType::kI16:
      case DType::kI32:
      case DType::kI64:
        return v.i64;
      case DType::kU8:
      case DType::kU16:
      case DType::kU32:
      case DType::kU64:
        return static_cast<int64_t>(v.u64);
      case DType::kF16:
      case DType::kBF16:
      case DType::kF32:
      case DType::kF64:
        return static_cast<int64_t>(v.f64);
      case DType::kDecimal:
        return static_cast<int64_t>(v.decimal);
      case DType::kRational:
        return static_cast<int64_t>(v.rational.num / v.rational.den);
      default:
        throw util::Error("Cannot cast complex/decimal/rational/function", 0, 0);
    }
  };
  auto as_unsigned = [&]() -> uint64_t {
    switch (v.type) {
      case DType::kBool:
        return v.boolean ? 1 : 0;
      case DType::kI8:
      case DType::kI16:
      case DType::kI32:
      case DType::kI64:
        return static_cast<uint64_t>(v.i64);
      case DType::kU8:
      case DType::kU16:
      case DType::kU32:
      case DType::kU64:
        return v.u64;
      case DType::kF16:
      case DType::kBF16:
      case DType::kF32:
      case DType::kF64:
        return static_cast<uint64_t>(v.f64);
      case DType::kDecimal:
        return static_cast<uint64_t>(v.decimal);
      case DType::kRational:
        return static_cast<uint64_t>(v.rational.num / v.rational.den);
      default:
        throw util::Error("Cannot cast complex/decimal/rational/function", 0, 0);
    }
  };
  switch (target) {
    case DType::kBool:
      return Value::Bool(as_double() != 0.0);
    case DType::kI8:
      return Value::I8(static_cast<int8_t>(as_signed()));
    case DType::kI16:
      return Value::I16(static_cast<int16_t>(as_signed()));
    case DType::kI32:
      return Value::I32(static_cast<int32_t>(as_signed()));
    case DType::kI64:
      return Value::I64(as_signed());
    case DType::kU8:
      return Value::U8(static_cast<uint8_t>(as_unsigned()));
    case DType::kU16:
      return Value::U16(static_cast<uint16_t>(as_unsigned()));
    case DType::kU32:
      return Value::U32(static_cast<uint32_t>(as_unsigned()));
    case DType::kU64:
      return Value::U64(as_unsigned());
    case DType::kF16:
      return Value::F16(static_cast<float>(as_double()));
    case DType::kBF16:
      return Value::BF16(static_cast<float>(as_double()));
    case DType::kF32:
      return Value::F32(static_cast<float>(as_double()));
    case DType::kF64:
      return Value::F64(as_double());
    case DType::kC64:
      if (v.type == DType::kC64) {
        return v;
      }
      if (v.type == DType::kC128) {
        return Value::Complex64(std::complex<float>(static_cast<float>(v.complex.real()),
                                                    static_cast<float>(v.complex.imag())));
      }
      return Value::Complex64(std::complex<float>(static_cast<float>(as_double()), 0.0f));
    case DType::kC128:
      if (v.type == DType::kC128) {
        return v;
      }
      if (v.type == DType::kC64) {
        return Value::Complex128(std::complex<double>(v.complex.real(), v.complex.imag()));
      }
      return Value::Complex128(std::complex<double>(as_double(), 0.0));
    case DType::kDecimal:
      return Value::Decimal(static_cast<long double>(as_double()));
    case DType::kRational: {
      if (v.type == DType::kRational) {
        return v;
      }
      int64_t num = static_cast<int64_t>(as_signed());
      return Value::RationalValueNormalized(num, 1);
    }
    default:
      throw util::Error("Unsupported cast target", 0, 0);
  }
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

  DType target = PromoteType(lhs.type, rhs.type);
  Value lcast = CastTo(target, lhs);
  Value rcast = CastTo(target, rhs);

  if (target == DType::kDecimal) {
    long double lv = lcast.decimal;
    long double rv = rcast.decimal;
    lv = RoundDecimal(lv);
    rv = RoundDecimal(rv);
    switch (expr.op) {
      case parser::BinaryOp::kAdd:
        return Value::Decimal(RoundDecimal(lv + rv));
      case parser::BinaryOp::kSub:
        return Value::Decimal(RoundDecimal(lv - rv));
      case parser::BinaryOp::kMul:
        return Value::Decimal(RoundDecimal(lv * rv));
      case parser::BinaryOp::kDiv:
        if (rv == 0.0L) throw util::Error("Division by zero", 0, 0);
        return Value::Decimal(RoundDecimal(lv / rv));
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

  if (target == DType::kRational) {
    auto simplify = [](int64_t num, int64_t den) -> Value {
      if (den == 0) throw util::Error("Division by zero", 0, 0);
      return Value::RationalValueNormalized(num, den);
    };
    int64_t ln = lcast.rational.num;
    int64_t ld = lcast.rational.den;
    int64_t rn = rcast.rational.num;
    int64_t rd = rcast.rational.den;
    switch (expr.op) {
      case parser::BinaryOp::kAdd:
        return simplify(ln * rd + rn * ld, ld * rd);
      case parser::BinaryOp::kSub:
        return simplify(ln * rd - rn * ld, ld * rd);
      case parser::BinaryOp::kMul:
        return simplify(ln * rn, ld * rd);
      case parser::BinaryOp::kDiv:
        return simplify(ln * rd, ld * rn);
      case parser::BinaryOp::kEq:
        return Value::Bool(ln * rd == rn * ld);
      case parser::BinaryOp::kNe:
        return Value::Bool(ln * rd != rn * ld);
      case parser::BinaryOp::kGt:
        return Value::Bool(ln * rd > rn * ld);
      case parser::BinaryOp::kGe:
        return Value::Bool(ln * rd >= rn * ld);
      case parser::BinaryOp::kLt:
        return Value::Bool(ln * rd < rn * ld);
      case parser::BinaryOp::kLe:
        return Value::Bool(ln * rd <= rn * ld);
    }
  }

  if (is_complex(target)) {
    std::complex<double> lcv =
        target == DType::kC128 ? lcast.complex
                               : std::complex<double>(static_cast<double>(lcast.complex.real()),
                                                      static_cast<double>(lcast.complex.imag()));
    std::complex<double> rcv =
        target == DType::kC128 ? rcast.complex
                               : std::complex<double>(static_cast<double>(rcast.complex.real()),
                                                      static_cast<double>(rcast.complex.imag()));
    switch (expr.op) {
      case parser::BinaryOp::kAdd: {
        auto res = lcv + rcv;
        return target == DType::kC64
                   ? Value::Complex64(std::complex<float>(static_cast<float>(res.real()),
                                                          static_cast<float>(res.imag())))
                   : Value::Complex128(res);
      }
      case parser::BinaryOp::kSub: {
        auto res = lcv - rcv;
        return target == DType::kC64
                   ? Value::Complex64(std::complex<float>(static_cast<float>(res.real()),
                                                          static_cast<float>(res.imag())))
                   : Value::Complex128(res);
      }
      case parser::BinaryOp::kMul: {
        auto res = lcv * rcv;
        return target == DType::kC64
                   ? Value::Complex64(std::complex<float>(static_cast<float>(res.real()),
                                                          static_cast<float>(res.imag())))
                   : Value::Complex128(res);
      }
      case parser::BinaryOp::kDiv: {
        auto res = lcv / rcv;
        return target == DType::kC64
                   ? Value::Complex64(std::complex<float>(static_cast<float>(res.real()),
                                                          static_cast<float>(res.imag())))
                   : Value::Complex128(res);
      }
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

  if (is_float(target)) {
    double lv = lcast.f64;
    double rv = rcast.f64;
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

  if (is_int(target)) {
    const bool is_unsigned = IsUnsignedInt(target);
    if (is_unsigned) {
      uint64_t lv = lcast.u64;
      uint64_t rv = rcast.u64;
      switch (expr.op) {
        case parser::BinaryOp::kAdd:
          return target == DType::kU32 ? Value::U32(static_cast<uint32_t>(lv + rv))
                                       : Value::U64(lv + rv);
        case parser::BinaryOp::kSub:
          return target == DType::kU32 ? Value::U32(static_cast<uint32_t>(lv - rv))
                                       : Value::U64(lv - rv);
        case parser::BinaryOp::kMul:
          return target == DType::kU32 ? Value::U32(static_cast<uint32_t>(lv * rv))
                                       : Value::U64(lv * rv);
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
    } else {
      int64_t lv = lcast.i64;
      int64_t rv = rcast.i64;
      switch (expr.op) {
        case parser::BinaryOp::kAdd:
          switch (target) {
            case DType::kI8:
              return Value::I8(static_cast<int8_t>(lv + rv));
            case DType::kI16:
              return Value::I16(static_cast<int16_t>(lv + rv));
            case DType::kI32:
              return Value::I32(static_cast<int32_t>(lv + rv));
            default:
              return Value::I64(lv + rv);
          }
        case parser::BinaryOp::kSub:
          switch (target) {
            case DType::kI8:
              return Value::I8(static_cast<int8_t>(lv - rv));
            case DType::kI16:
              return Value::I16(static_cast<int16_t>(lv - rv));
            case DType::kI32:
              return Value::I32(static_cast<int32_t>(lv - rv));
            default:
              return Value::I64(lv - rv);
          }
        case parser::BinaryOp::kMul:
          switch (target) {
            case DType::kI8:
              return Value::I8(static_cast<int8_t>(lv * rv));
            case DType::kI16:
              return Value::I16(static_cast<int16_t>(lv * rv));
            case DType::kI32:
              return Value::I32(static_cast<int32_t>(lv * rv));
            default:
              return Value::I64(lv * rv);
          }
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
  // Handle cast before evaluating all args to allow type-name first arg.
  if (call.callee == "cast") {
    if (call.args.size() != 2) {
      throw util::Error("cast expects two arguments: type name and expression", 0, 0);
    }
    const auto* type_id = dynamic_cast<const parser::Identifier*>(call.args[0].get());
    if (type_id == nullptr) {
      throw util::Error("cast first argument must be a type name identifier", 0, 0);
    }
    auto dt = LookupDType(type_id->name);
    if (!dt.has_value()) {
      throw util::Error("Unknown cast target type: " + type_id->name, 0, 0);
    }
    Value v = Evaluate(*call.args[1]);
    return CastTo(dt.value(), v);
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

  if (name == "set_decimal_precision") {
    expect_args(1, name);
    int p = static_cast<int>(args[0].f64);
    SetDecimalPrecision(p);
    return Value::Number(0.0);
  }
  if (name == "get_decimal_precision") {
    expect_args(0, name);
    return Value::I32(GetDecimalPrecision());
  }
  if (name == "decimal") {
    expect_args(1, name);
    long double v = static_cast<long double>(args[0].f64);
    return Value::Decimal(RoundDecimal(v));
  }
  if (name == "rational") {
    expect_args(2, name);
    int64_t num = static_cast<int64_t>(args[0].f64);
    int64_t den = static_cast<int64_t>(args[1].f64);
    if (den == 0) {
      throw util::Error("rational denominator cannot be zero", 0, 0);
    }
    return Value::RationalValueNormalized(num, den);
  }
  if (name == "complex") {
    expect_args(2, name);
    double re = args[0].f64;
    double im = args[1].f64;
    return Value::Complex128(std::complex<double>(re, im));
  }
  if (name == "abs") {
    expect_args(1, name);
    if (args[0].type == DType::kC64 || args[0].type == DType::kC128) {
      return Value::F64(std::abs(args[0].complex));
    }
    return Value::F64(std::fabs(args[0].f64));
  }

  if (name == "pow") {
    expect_args(2, name);
    if ((args[0].type == DType::kC64 || args[0].type == DType::kC128) ||
        (args[1].type == DType::kC64 || args[1].type == DType::kC128)) {
      std::complex<double> base =
          args[0].type == DType::kC128 ? args[0].complex : std::complex<double>(args[0].f64, 0.0);
      std::complex<double> expv =
          args[1].type == DType::kC128 ? args[1].complex : std::complex<double>(args[1].f64, 0.0);
      auto res = std::pow(base, expv);
      return Value::Complex128(res);
    }
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
    case DType::kF64: {
      std::ostringstream oss;
      int precision = 6;
      if (type == DType::kF32) precision = 7;
      if (type == DType::kF64) precision = 15;
      oss << std::setprecision(precision) << f64;
      return oss.str();
    }
    case DType::kC64:
    case DType::kC128: {
      std::ostringstream oss;
      oss << std::setprecision(10) << complex.real() << "+" << complex.imag() << "i";
      return oss.str();
    }
    case DType::kDecimal: {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(GetDecimalPrecision())
          << static_cast<double>(RoundDecimal(decimal));
      return oss.str();
    }
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
