#include "runtime/ops.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
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

void EnsureFiniteOrThrow(const Value& v, int line, int column) {
  auto finite = [](double d) { return std::isfinite(d); };
  switch (v.type) {
    case DType::kF16:
    case DType::kBF16:
    case DType::kF32:
    case DType::kF64:
      if (!finite(v.f64)) {
        throw util::Error("Non-finite result (NaN/inf)", line, column);
      }
      break;
    case DType::kC64:
    case DType::kC128:
      if (!finite(v.complex.real()) || !finite(v.complex.imag())) {
        throw util::Error("Non-finite complex result (NaN/inf)", line, column);
      }
      break;
    case DType::kTensor: {
      const double* data = v.tensor.Data();
      for (int64_t i = 0; i < v.tensor.size; ++i) {
        if (!finite(data[i])) {
          throw util::Error("Tensor contains non-finite value (NaN/inf)", line, column);
        }
      }
      break;
    }
    default:
      break;
  }
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
std::string ShapeToString(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) oss << "x";
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

std::optional<std::vector<int64_t>> BroadcastShape(const std::vector<int64_t>& a,
                                                   const std::vector<int64_t>& b) {
  const size_t out_rank = std::max(a.size(), b.size());
  std::vector<int64_t> result(out_rank, 1);
  for (size_t i = 0; i < out_rank; ++i) {
    const bool a_has_dim = a.size() > i;
    const bool b_has_dim = b.size() > i;
    const int64_t a_dim = a_has_dim ? a[a.size() - 1 - i] : 1;
    const int64_t b_dim = b_has_dim ? b[b.size() - 1 - i] : 1;
    if (a_dim == b_dim || a_dim == 1 || b_dim == 1) {
      result[out_rank - 1 - i] = std::max(a_dim, b_dim);
    } else {
      return std::nullopt;
    }
  }
  return result;
}

std::vector<int64_t> BroadcastStrides(const std::vector<int64_t>& shape,
                                      const std::vector<int64_t>& strides, size_t out_rank) {
  std::vector<int64_t> out(out_rank, 0);
  const size_t offset = out_rank - shape.size();
  for (size_t i = 0; i < shape.size(); ++i) {
    out[offset + i] = (shape[i] == 1) ? 0 : strides[i];
  }
  return out;
}

int64_t OffsetFromFlatIndex(int64_t flat, const std::vector<int64_t>& out_strides,
                            const std::vector<int64_t>& broadcast_strides) {
  int64_t offset = 0;
  int64_t idx = flat;
  for (size_t dim = 0; dim < out_strides.size(); ++dim) {
    const int64_t stride = out_strides[dim];
    const int64_t coord = stride == 0 ? 0 : idx / stride;
    idx -= coord * stride;
    offset += coord * broadcast_strides[dim];
  }
  return offset;
}
std::string DTypeToString(DType t) {
  switch (t) {
    case DType::kBool:
      return "bool";
    case DType::kI8:
      return "i8";
    case DType::kI16:
      return "i16";
    case DType::kI32:
      return "i32";
    case DType::kI64:
      return "i64";
    case DType::kU8:
      return "u8";
    case DType::kU16:
      return "u16";
    case DType::kU32:
      return "u32";
    case DType::kU64:
      return "u64";
    case DType::kF16:
      return "f16";
    case DType::kBF16:
      return "bfloat16";
    case DType::kF32:
      return "f32";
    case DType::kF64:
      return "f64";
    case DType::kC64:
      return "complex64";
    case DType::kC128:
      return "complex128";
    case DType::kString:
      return "string";
    case DType::kDecimal:
      return "decimal";
    case DType::kRational:
      return "rational";
    case DType::kFunction:
      return "function";
    case DType::kTensor:
      return "tensor";
    case DType::kTuple:
      return "tuple";
    case DType::kRecord:
      return "record";
  }
  return "";
}

bool ValueMatchesType(const Value& value, const std::string& type_name) {
  if (type_name.empty()) {
    return true;
  }
  if (IsBoolTypeName(type_name)) {
    return value.type == DType::kBool;
  }
  if (type_name == "string") {
    return value.type == DType::kString;
  }
  if (IsNumericTypeName(type_name)) {
    if (value.type == DType::kBool || value.type == DType::kFunction) return false;
    if (value.type_name.empty()) return true;
    if (IsNumericTypeName(value.type_name)) return true;
    return value.type_name == type_name;
  }
  if (type_name == "tensor") {
    return value.type == DType::kTensor;
  }
  // For now, unsupported or unknown types fail.
  return false;
}

std::string ValueTypeName(const Value& v) {
  if (!v.type_name.empty()) return v.type_name;
  return DTypeToString(v.type);
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
  if (a == DType::kString || b == DType::kString) {
    if (a == b) return DType::kString;
    throw util::Error("String can only operate with string", 0, 0);
  }
  if (a == DType::kTensor || b == DType::kTensor) {
    return DType::kTensor;
  }
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
        throw util::Error("Cannot cast complex/decimal/rational/string/function", 0, 0);
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
    case DType::kString:
      if (v.type == DType::kString) return v;
      throw util::Error("Cannot cast to string", 0, 0);
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
  if (const auto* str_lit = dynamic_cast<const parser::StringLiteral*>(&expr)) {
    return Value::String(str_lit->value);
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
  if (const auto* tuple = dynamic_cast<const parser::TupleLiteral*>(&expr)) {
    std::vector<Value> elems;
    elems.reserve(tuple->elements.size());
    for (const auto& e : tuple->elements) {
      elems.push_back(Evaluate(*e));
    }
    return Value::Tuple(std::move(elems));
  }
  if (const auto* record = dynamic_cast<const parser::RecordLiteral*>(&expr)) {
    std::vector<std::pair<std::string, Value>> fields;
    fields.reserve(record->fields.size());
    for (const auto& f : record->fields) {
      fields.emplace_back(f.first, Evaluate(*f.second));
    }
    return Value::Record(std::move(fields));
  }
  if (const auto* idx = dynamic_cast<const parser::IndexExpression*>(&expr)) {
    Value object = Evaluate(*idx->object);
    Value index = Evaluate(*idx->index);
    if (object.type == DType::kTuple) {
      int64_t pos = static_cast<int64_t>(index.f64);
      if (pos < 0 || pos >= static_cast<int64_t>(object.tuple.elements.size())) {
        throw util::Error("Tuple index out of range", idx->line, idx->column);
      }
      return object.tuple.elements[static_cast<size_t>(pos)];
    } else if (object.type == DType::kRecord) {
      if (index.type != DType::kString) {
        throw util::Error("Record index must be a string", idx->line, idx->column);
      }
      auto it = object.record.index.find(index.str);
      if (it == object.record.index.end()) {
        throw util::Error("Record key not found: " + index.str, idx->line, idx->column);
      }
      return object.record.fields[it->second].second;
    }
    throw util::Error("Indexing not supported on this type", idx->line, idx->column);
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
  auto ensure = [&](Value v) {
    EnsureFiniteOrThrow(v, expr.line, expr.column);
    return v;
  };
  switch (expr.op) {
    case parser::UnaryOp::kNegate:
      switch (operand.type) {
        case DType::kBool:
          return ensure(Value::Bool(!operand.boolean));
        case DType::kI32:
        case DType::kI64:
          return ensure(Value::I64(-operand.i64));
        case DType::kU32:
        case DType::kU64:
          return ensure(Value::I64(-static_cast<int64_t>(operand.u64)));
        case DType::kF32:
          return ensure(Value::F32(-static_cast<float>(operand.f64)));
        case DType::kF64:
          return ensure(Value::F64(-operand.f64));
        default:
          throw util::Error("Unary negate not supported for type " + ValueTypeName(operand), 0, 0);
      }
  }
  throw std::runtime_error("Unhandled unary operator");
}

Value Evaluator::EvaluateBinary(const parser::BinaryExpression& expr) {
  Value lhs = Evaluate(*expr.lhs);
  Value rhs = Evaluate(*expr.rhs);
  auto ensure = [&](Value v) {
    EnsureFiniteOrThrow(v, expr.line, expr.column);
    return v;
  };
  // Hot path for identical common types.
  if (lhs.type == rhs.type) {
    switch (lhs.type) {
      case DType::kI32: {
        int32_t lv = static_cast<int32_t>(lhs.i64);
        int32_t rv = static_cast<int32_t>(rhs.i64);
        switch (expr.op) {
          case parser::BinaryOp::kAdd:
            return ensure(Value::I32(lv + rv));
          case parser::BinaryOp::kSub:
            return ensure(Value::I32(lv - rv));
          case parser::BinaryOp::kMul:
            return ensure(Value::I32(lv * rv));
          case parser::BinaryOp::kDiv:
            if (rv == 0) {
              throw util::Error("Division by zero", expr.line, expr.column);
            }
            return ensure(Value::F64(static_cast<double>(lv) / static_cast<double>(rv)));
          case parser::BinaryOp::kEq:
            return ensure(Value::Bool(lv == rv));
          case parser::BinaryOp::kNe:
            return ensure(Value::Bool(lv != rv));
          case parser::BinaryOp::kGt:
            return ensure(Value::Bool(lv > rv));
          case parser::BinaryOp::kGe:
            return ensure(Value::Bool(lv >= rv));
          case parser::BinaryOp::kLt:
            return ensure(Value::Bool(lv < rv));
          case parser::BinaryOp::kLe:
            return ensure(Value::Bool(lv <= rv));
        }
      } break;
      case DType::kI64: {
        int64_t lv = lhs.i64;
        int64_t rv = rhs.i64;
        switch (expr.op) {
          case parser::BinaryOp::kAdd:
            return ensure(Value::I64(lv + rv));
          case parser::BinaryOp::kSub:
            return ensure(Value::I64(lv - rv));
          case parser::BinaryOp::kMul:
            return ensure(Value::I64(lv * rv));
          case parser::BinaryOp::kDiv:
            if (rv == 0) {
              throw util::Error("Division by zero", expr.line, expr.column);
            }
            return ensure(Value::F64(static_cast<double>(lv) / static_cast<double>(rv)));
          case parser::BinaryOp::kEq:
            return ensure(Value::Bool(lv == rv));
          case parser::BinaryOp::kNe:
            return ensure(Value::Bool(lv != rv));
          case parser::BinaryOp::kGt:
            return ensure(Value::Bool(lv > rv));
          case parser::BinaryOp::kGe:
            return ensure(Value::Bool(lv >= rv));
          case parser::BinaryOp::kLt:
            return ensure(Value::Bool(lv < rv));
          case parser::BinaryOp::kLe:
            return ensure(Value::Bool(lv <= rv));
        }
      } break;
      case DType::kF64: {
        double lv = lhs.f64;
        double rv = rhs.f64;
        switch (expr.op) {
          case parser::BinaryOp::kAdd:
            return ensure(Value::F64(lv + rv));
          case parser::BinaryOp::kSub:
            return ensure(Value::F64(lv - rv));
          case parser::BinaryOp::kMul:
            return ensure(Value::F64(lv * rv));
          case parser::BinaryOp::kDiv:
            if (rv == 0.0) {
              throw util::Error("Division by zero", expr.line, expr.column);
            }
            return ensure(Value::F64(lv / rv));
          case parser::BinaryOp::kEq:
            return ensure(Value::Bool(lv == rv));
          case parser::BinaryOp::kNe:
            return ensure(Value::Bool(lv != rv));
          case parser::BinaryOp::kGt:
            return ensure(Value::Bool(lv > rv));
          case parser::BinaryOp::kGe:
            return ensure(Value::Bool(lv >= rv));
          case parser::BinaryOp::kLt:
            return ensure(Value::Bool(lv < rv));
          case parser::BinaryOp::kLe:
            return ensure(Value::Bool(lv <= rv));
        }
      } break;
      case DType::kC128: {
        const auto& lv = lhs.complex;
        const auto& rv = rhs.complex;
        switch (expr.op) {
          case parser::BinaryOp::kAdd:
            return ensure(Value::Complex128(lv + rv));
          case parser::BinaryOp::kSub:
            return ensure(Value::Complex128(lv - rv));
          case parser::BinaryOp::kMul:
            return ensure(Value::Complex128(lv * rv));
          case parser::BinaryOp::kDiv:
            if (rv.real() == 0.0 && rv.imag() == 0.0) {
              throw util::Error("Division by zero", expr.line, expr.column);
            }
            return ensure(Value::Complex128(lv / rv));
          case parser::BinaryOp::kEq:
            return ensure(Value::Bool(lv == rv));
          case parser::BinaryOp::kNe:
            return ensure(Value::Bool(lv != rv));
          case parser::BinaryOp::kGt:
          case parser::BinaryOp::kGe:
          case parser::BinaryOp::kLt:
          case parser::BinaryOp::kLe:
            throw util::Error("Complex comparison not supported", expr.line, expr.column);
        }
      } break;
      case DType::kString: {
        switch (expr.op) {
          case parser::BinaryOp::kEq:
            return ensure(Value::Bool(lhs.str == rhs.str));
          case parser::BinaryOp::kNe:
            return ensure(Value::Bool(lhs.str != rhs.str));
          default:
            throw util::Error("Only equality/inequality supported for strings", expr.line,
                              expr.column);
        }
      } break;
      default:
        break;
    }
  }
  if (lhs.type == DType::kTensor || rhs.type == DType::kTensor) {
    const std::vector<int64_t> lhs_shape =
        lhs.type == DType::kTensor ? lhs.tensor.shape : std::vector<int64_t>{};
    const std::vector<int64_t> rhs_shape =
        rhs.type == DType::kTensor ? rhs.tensor.shape : std::vector<int64_t>{};
    auto broadcast_shape = BroadcastShape(lhs_shape, rhs_shape);
    if (!broadcast_shape.has_value()) {
      throw util::Error("Tensor shapes are not broadcastable (lhs " + ShapeToString(lhs_shape) +
                            ", rhs " + ShapeToString(rhs_shape) + ")",
                        0, 0);
    }
    const std::vector<int64_t>& out_shape = *broadcast_shape;
    DType elem_target;
    if (lhs.type == DType::kTensor && rhs.type == DType::kTensor) {
      elem_target = PromoteType(lhs.tensor.elem_type, rhs.tensor.elem_type);
    } else if (lhs.type == DType::kTensor) {
      elem_target = PromoteType(lhs.tensor.elem_type, rhs.type);
    } else {
      elem_target = PromoteType(rhs.tensor.elem_type, lhs.type);
    }
    Value out = Value::Tensor(out_shape, elem_target, 0.0);
    std::vector<int64_t> lhs_bstrides =
        lhs.type == DType::kTensor
            ? BroadcastStrides(lhs.tensor.shape, lhs.tensor.strides, out_shape.size())
            : std::vector<int64_t>(out_shape.size(), 0);
    std::vector<int64_t> rhs_bstrides =
        rhs.type == DType::kTensor
            ? BroadcastStrides(rhs.tensor.shape, rhs.tensor.strides, out_shape.size())
            : std::vector<int64_t>(out_shape.size(), 0);

    auto lhs_data = [&](int64_t idx) -> double {
      if (lhs.type != DType::kTensor) return lhs.f64;
      const int64_t offset = OffsetFromFlatIndex(idx, out.tensor.strides, lhs_bstrides);
      return lhs.tensor.Data()[offset];
    };
    auto rhs_data = [&](int64_t idx) -> double {
      if (rhs.type != DType::kTensor) return rhs.f64;
      const int64_t offset = OffsetFromFlatIndex(idx, out.tensor.strides, rhs_bstrides);
      return rhs.tensor.Data()[offset];
    };

    for (int64_t i = 0; i < out.tensor.size; ++i) {
      double lv = lhs_data(i);
      double rv = rhs_data(i);
      switch (expr.op) {
        case parser::BinaryOp::kAdd:
          out.tensor.Data()[i] = lv + rv;
          break;
        case parser::BinaryOp::kSub:
          out.tensor.Data()[i] = lv - rv;
          break;
        case parser::BinaryOp::kMul:
          out.tensor.Data()[i] = lv * rv;
          break;
        case parser::BinaryOp::kDiv:
          if (rv == 0.0) {
            throw util::Error("Division by zero", expr.line, expr.column);
          }
          out.tensor.Data()[i] = lv / rv;
          break;
        default:
          throw util::Error("Tensor comparison not supported", 0, 0);
      }
    }
    EnsureFiniteOrThrow(out, expr.line, expr.column);
    return out;
  }
  if (lhs.type == DType::kTuple && rhs.type == DType::kTuple) {
    if (expr.op == parser::BinaryOp::kEq || expr.op == parser::BinaryOp::kNe) {
      bool equal = lhs.tuple.elements.size() == rhs.tuple.elements.size();
      if (equal) {
        for (size_t i = 0; i < lhs.tuple.elements.size(); ++i) {
          if (lhs.tuple.elements[i].ToString() != rhs.tuple.elements[i].ToString()) {
            equal = false;
            break;
          }
        }
      }
      return Value::Bool(expr.op == parser::BinaryOp::kEq ? equal : !equal);
    }
    throw util::Error("Tuple comparison only supports == and !=", 0, 0);
  }
  if (lhs.type == DType::kRecord && rhs.type == DType::kRecord) {
    if (expr.op == parser::BinaryOp::kEq || expr.op == parser::BinaryOp::kNe) {
      bool equal = lhs.record.fields.size() == rhs.record.fields.size();
      if (equal) {
        for (size_t i = 0; i < lhs.record.fields.size(); ++i) {
          if (lhs.record.fields[i].first != rhs.record.fields[i].first ||
              lhs.record.fields[i].second.ToString() != rhs.record.fields[i].second.ToString()) {
            equal = false;
            break;
          }
        }
      }
      return Value::Bool(expr.op == parser::BinaryOp::kEq ? equal : !equal);
    }
    throw util::Error("Record comparison only supports == and !=", 0, 0);
  }
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
        return ensure(target == DType::kC64
                          ? Value::Complex64(std::complex<float>(static_cast<float>(res.real()),
                                                                 static_cast<float>(res.imag())))
                          : Value::Complex128(res));
      }
      case parser::BinaryOp::kSub: {
        auto res = lcv - rcv;
        return ensure(target == DType::kC64
                          ? Value::Complex64(std::complex<float>(static_cast<float>(res.real()),
                                                                 static_cast<float>(res.imag())))
                          : Value::Complex128(res));
      }
      case parser::BinaryOp::kMul: {
        auto res = lcv * rcv;
        return ensure(target == DType::kC64
                          ? Value::Complex64(std::complex<float>(static_cast<float>(res.real()),
                                                                 static_cast<float>(res.imag())))
                          : Value::Complex128(res));
      }
      case parser::BinaryOp::kDiv: {
        if (rcv.real() == 0.0 && rcv.imag() == 0.0) {
          throw util::Error("Division by zero", expr.line, expr.column);
        }
        auto res = lcv / rcv;
        return ensure(target == DType::kC64
                          ? Value::Complex64(std::complex<float>(static_cast<float>(res.real()),
                                                                 static_cast<float>(res.imag())))
                          : Value::Complex128(res));
      }
      case parser::BinaryOp::kEq:
        return ensure(Value::Bool(lcv == rcv));
      case parser::BinaryOp::kNe:
        return ensure(Value::Bool(lcv != rcv));
      case parser::BinaryOp::kGt:
      case parser::BinaryOp::kGe:
      case parser::BinaryOp::kLt:
      case parser::BinaryOp::kLe:
        throw util::Error("Complex comparison not supported", expr.line, expr.column);
    }
  }

  if (is_float(target)) {
    double lv = lcast.f64;
    double rv = rcast.f64;
    switch (expr.op) {
      case parser::BinaryOp::kAdd:
        return ensure(Value::F64(lv + rv));
      case parser::BinaryOp::kSub:
        return ensure(Value::F64(lv - rv));
      case parser::BinaryOp::kMul:
        return ensure(Value::F64(lv * rv));
      case parser::BinaryOp::kDiv:
        if (rv == 0.0) {
          throw util::Error("Division by zero", expr.line, expr.column);
        }
        return ensure(Value::F64(lv / rv));
      case parser::BinaryOp::kEq:
        return ensure(Value::Bool(lv == rv));
      case parser::BinaryOp::kNe:
        return ensure(Value::Bool(lv != rv));
      case parser::BinaryOp::kGt:
        return ensure(Value::Bool(lv > rv));
      case parser::BinaryOp::kGe:
        return ensure(Value::Bool(lv >= rv));
      case parser::BinaryOp::kLt:
        return ensure(Value::Bool(lv < rv));
      case parser::BinaryOp::kLe:
        return ensure(Value::Bool(lv <= rv));
    }
  }

  if (is_int(target)) {
    const bool is_unsigned = IsUnsignedInt(target);
    if (is_unsigned) {
      uint64_t lv = lcast.u64;
      uint64_t rv = rcast.u64;
      switch (expr.op) {
        case parser::BinaryOp::kAdd:
          return ensure(target == DType::kU32 ? Value::U32(static_cast<uint32_t>(lv + rv))
                                              : Value::U64(lv + rv));
        case parser::BinaryOp::kSub:
          return ensure(target == DType::kU32 ? Value::U32(static_cast<uint32_t>(lv - rv))
                                              : Value::U64(lv - rv));
        case parser::BinaryOp::kMul:
          return ensure(target == DType::kU32 ? Value::U32(static_cast<uint32_t>(lv * rv))
                                              : Value::U64(lv * rv));
        case parser::BinaryOp::kDiv:
          if (rv == 0) {
            throw util::Error("Division by zero", expr.line, expr.column);
          }
          return ensure(Value::F64(static_cast<double>(lv) / static_cast<double>(rv)));
        case parser::BinaryOp::kEq:
          return ensure(Value::Bool(lv == rv));
        case parser::BinaryOp::kNe:
          return ensure(Value::Bool(lv != rv));
        case parser::BinaryOp::kGt:
          return ensure(Value::Bool(lv > rv));
        case parser::BinaryOp::kGe:
          return ensure(Value::Bool(lv >= rv));
        case parser::BinaryOp::kLt:
          return ensure(Value::Bool(lv < rv));
        case parser::BinaryOp::kLe:
          return ensure(Value::Bool(lv <= rv));
      }
    } else {
      int64_t lv = lcast.i64;
      int64_t rv = rcast.i64;
      switch (expr.op) {
        case parser::BinaryOp::kAdd:
          switch (target) {
            case DType::kI8:
              return ensure(Value::I8(static_cast<int8_t>(lv + rv)));
            case DType::kI16:
              return ensure(Value::I16(static_cast<int16_t>(lv + rv)));
            case DType::kI32:
              return ensure(Value::I32(static_cast<int32_t>(lv + rv)));
            default:
              return ensure(Value::I64(lv + rv));
          }
        case parser::BinaryOp::kSub:
          switch (target) {
            case DType::kI8:
              return ensure(Value::I8(static_cast<int8_t>(lv - rv)));
            case DType::kI16:
              return ensure(Value::I16(static_cast<int16_t>(lv - rv)));
            case DType::kI32:
              return ensure(Value::I32(static_cast<int32_t>(lv - rv)));
            default:
              return ensure(Value::I64(lv - rv));
          }
        case parser::BinaryOp::kMul:
          switch (target) {
            case DType::kI8:
              return ensure(Value::I8(static_cast<int8_t>(lv * rv)));
            case DType::kI16:
              return ensure(Value::I16(static_cast<int16_t>(lv * rv)));
            case DType::kI32:
              return ensure(Value::I32(static_cast<int32_t>(lv * rv)));
            default:
              return ensure(Value::I64(lv * rv));
          }
        case parser::BinaryOp::kDiv:
          if (rv == 0) {
            throw util::Error("Division by zero", expr.line, expr.column);
          }
          return ensure(Value::F64(static_cast<double>(lv) / static_cast<double>(rv)));
        case parser::BinaryOp::kEq:
          return ensure(Value::Bool(lv == rv));
        case parser::BinaryOp::kNe:
          return ensure(Value::Bool(lv != rv));
        case parser::BinaryOp::kGt:
          return ensure(Value::Bool(lv > rv));
        case parser::BinaryOp::kGe:
          return ensure(Value::Bool(lv >= rv));
        case parser::BinaryOp::kLt:
          return ensure(Value::Bool(lv < rv));
        case parser::BinaryOp::kLe:
          return ensure(Value::Bool(lv <= rv));
      }
    }
  }

  // Fallback to double.
  double lv = lhs.f64;
  double rv = rhs.f64;
  switch (expr.op) {
    case parser::BinaryOp::kAdd:
      return ensure(Value::F64(lv + rv));
    case parser::BinaryOp::kSub:
      return ensure(Value::F64(lv - rv));
    case parser::BinaryOp::kMul:
      return ensure(Value::F64(lv * rv));
    case parser::BinaryOp::kDiv:
      if (rv == 0.0) {
        throw util::Error("Division by zero", expr.line, expr.column);
      }
      return ensure(Value::F64(lv / rv));
    case parser::BinaryOp::kEq:
      return ensure(Value::Bool(lv == rv));
    case parser::BinaryOp::kNe:
      return ensure(Value::Bool(lv != rv));
    case parser::BinaryOp::kGt:
      return ensure(Value::Bool(lv > rv));
    case parser::BinaryOp::kGe:
      return ensure(Value::Bool(lv >= rv));
    case parser::BinaryOp::kLt:
      return ensure(Value::Bool(lv < rv));
    case parser::BinaryOp::kLe:
      return ensure(Value::Bool(lv <= rv));
  }
  throw std::runtime_error("Unhandled binary operator");
}

Value Evaluator::EvaluateIdentifier(const parser::Identifier& identifier) {
  if (env_ == nullptr) {
    throw util::Error("Environment is not configured", identifier.line, identifier.column);
  }
  auto value = env_->Get(identifier.name);
  if (!value.has_value()) {
    throw util::Error("Undefined identifier: " + identifier.name, identifier.line,
                      identifier.column);
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
      throw util::Error("cast expects two arguments: type name and expression", call.line,
                        call.column);
    }
    const auto* type_id = dynamic_cast<const parser::Identifier*>(call.args[0].get());
    if (type_id == nullptr) {
      throw util::Error("cast first argument must be a type name identifier", call.line,
                        call.column);
    }
    auto dt = LookupDType(type_id->name);
    if (!dt.has_value()) {
      throw util::Error("Unknown cast target type: " + type_id->name, call.line, call.column);
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
      throw util::Error("Function is null: " + name, call.line, call.column);
    }
    if (fn->body == nullptr) {
      if (name == "print") {
        if (args.size() != 1) {
          throw util::Error("print expects 1 argument", call.line, call.column);
        }
        std::cout << args[0].ToString() << "\n";
        return Value::Number(0.0);
      }
      throw util::Error("Function has no body: " + name, call.line, call.column);
    }
    if (fn->parameters.size() != args.size()) {
      throw util::Error(name + " expects " + std::to_string(fn->parameters.size()) + " arguments",
                        call.line, call.column);
    }
    Environment fn_env(fn->defining_env);
    for (size_t i = 0; i < args.size(); ++i) {
      if (!fn->parameter_types.empty() && i < fn->parameter_types.size()) {
        const std::string& annot = fn->parameter_types[i];
        if (!annot.empty() && !ValueMatchesType(args[i], annot)) {
          throw util::Error("Type mismatch for parameter '" + fn->parameters[i] + "' (expected " +
                                annot + ", got " + ValueTypeName(args[i]) + ")",
                            call.line, call.column);
        }
      }
      fn_env.Define(fn->parameters[i], args[i]);
    }
    Evaluator fn_evaluator(&fn_env);
    ExecResult body_result = fn_evaluator.EvaluateStatement(*fn->body);
    if (!fn->return_type.empty() && body_result.value.has_value()) {
      if (!ValueMatchesType(body_result.value.value(), fn->return_type)) {
        throw util::Error("Return type mismatch in function " + name + " (expected " +
                              fn->return_type + ", got " +
                              ValueTypeName(body_result.value.value()) + ")",
                          call.line, call.column);
      }
      body_result.value->type_name = fn->return_type;
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
  auto ensure_not_tensor = [&](const Value& v, const std::string& func) {
    if (v.type == DType::kTensor) {
      throw util::Error(func + " does not support tensor arguments", 0, 0);
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
  if (name == "int") {
    expect_args(1, name);
    return CastTo(DType::kI64, args[0]);
  }
  if (name == "float") {
    expect_args(1, name);
    return CastTo(DType::kF64, args[0]);
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
  if (name == "tensor") {
    if (args.size() < 2) {
      throw util::Error("tensor expects at least shape and fill value", 0, 0);
    }
    std::vector<int64_t> shape;
    shape.reserve(args.size() - 1);
    for (size_t i = 0; i + 1 < args.size(); ++i) {
      shape.push_back(static_cast<int64_t>(args[i].f64));
    }
    double fill = args.back().f64;
    if (shape.empty()) {
      throw util::Error("tensor shape cannot be empty", 0, 0);
    }
    for (int64_t dim : shape) {
      if (dim <= 0) {
        throw util::Error("tensor dimensions must be positive (got " + std::to_string(dim) + ")", 0,
                          0);
      }
    }
    DType elem_type = DType::kF64;
    return Value::Tensor(shape, elem_type, fill);
  }
  if (name == "tensor_values") {
    if (args.empty()) {
      throw util::Error("tensor_values expects at least one value", 0, 0);
    }
    for (const auto& v : args) {
      if (v.type == DType::kTensor) {
        throw util::Error("tensor_values does not accept tensor arguments", 0, 0);
      }
    }
    DType elem_type = args[0].type;
    for (size_t i = 1; i < args.size(); ++i) {
      elem_type = PromoteType(elem_type, args[i].type);
    }
    Value out = Value::Tensor({static_cast<int64_t>(args.size())}, elem_type, 0.0);
    for (size_t i = 0; i < args.size(); ++i) {
      out.tensor.Data()[i] = CastTo(elem_type, args[i]).f64;
    }
    out.tensor.elem_type = elem_type;
    return out;
  }
  if (name == "pow") {
    expect_args(2, name);
    ensure_not_tensor(args[0], name);
    ensure_not_tensor(args[1], name);
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
    ensure_not_tensor(args[0], name);
    ensure_not_tensor(args[1], name);
    if (IsComplex(args[0].type) || IsComplex(args[1].type) || IsFloat(args[0].type) ||
        IsFloat(args[1].type) || args[0].type == DType::kDecimal ||
        args[0].type == DType::kRational || args[1].type == DType::kDecimal ||
        args[1].type == DType::kRational) {
      throw util::Error("gcd requires integer arguments", 0, 0);
    }
    auto a = static_cast<long long>(CastTo(DType::kI64, args[0]).i64);
    auto b = static_cast<long long>(CastTo(DType::kI64, args[1]).i64);
    return Value::I64(std::gcd(a, b));
  }
  if (name == "lcm") {
    expect_args(2, name);
    ensure_not_tensor(args[0], name);
    ensure_not_tensor(args[1], name);
    if (IsComplex(args[0].type) || IsComplex(args[1].type) || IsFloat(args[0].type) ||
        IsFloat(args[1].type) || args[0].type == DType::kDecimal ||
        args[0].type == DType::kRational || args[1].type == DType::kDecimal ||
        args[1].type == DType::kRational) {
      throw util::Error("lcm requires integer arguments", 0, 0);
    }
    auto a = static_cast<long long>(CastTo(DType::kI64, args[0]).i64);
    auto b = static_cast<long long>(CastTo(DType::kI64, args[1]).i64);
    return Value::I64(std::lcm(a, b));
  }
  if (name == "abs") {
    expect_args(1, name);
    ensure_not_tensor(args[0], name);
    switch (args[0].type) {
      case DType::kC64:
      case DType::kC128:
        return Value::F64(std::abs(args[0].complex));
      case DType::kDecimal:
        return Value::Decimal(std::fabsl(args[0].decimal));
      case DType::kRational:
        return Value::RationalValueNormalized(std::abs(args[0].rational.num),
                                              std::abs(args[0].rational.den));
      case DType::kI8:
      case DType::kI16:
      case DType::kI32:
      case DType::kI64:
        return Value::I64(std::abs(args[0].i64));
      case DType::kU8:
      case DType::kU16:
      case DType::kU32:
      case DType::kU64:
        return args[0];
      default:
        return Value::F64(std::fabs(args[0].f64));
    }
  }
  if (name == "sign") {
    expect_args(1, name);
    ensure_not_tensor(args[0], name);
    double v = args[0].f64;
    if (v > 0) return Value::I32(1);
    if (v < 0) return Value::I32(-1);
    return Value::I32(0);
  }
  if (name == "mod") {
    expect_args(2, name);
    ensure_not_tensor(args[0], name);
    ensure_not_tensor(args[1], name);
    if (args[1].f64 == 0.0) {
      throw util::Error("mod divisor cannot be zero", 0, 0);
    }
    if (!IsFloat(args[0].type) && !IsFloat(args[1].type) && !IsComplex(args[0].type) &&
        !IsComplex(args[1].type) && args[0].type != DType::kDecimal &&
        args[0].type != DType::kRational && args[1].type != DType::kDecimal &&
        args[1].type != DType::kRational) {
      auto lhs = CastTo(DType::kI64, args[0]).i64;
      auto rhs = CastTo(DType::kI64, args[1]).i64;
      return Value::I64(lhs % rhs);
    }
    return Value::F64(std::fmod(args[0].f64, args[1].f64));
  }
  if (name == "sum") {
    expect_args(1, name);
    const Value& v = args[0];
    if (v.type != DType::kTensor) {
      return Value::F64(v.f64);
    }
    double total = 0.0;
    for (int64_t i = 0; i < v.tensor.size; ++i) {
      total += v.tensor.Data()[i];
    }
    return Value::F64(total);
  }
  if (name == "mean") {
    expect_args(1, name);
    const Value& v = args[0];
    if (v.type != DType::kTensor) {
      return Value::F64(v.f64);
    }
    double total = 0.0;
    for (int64_t i = 0; i < v.tensor.size; ++i) {
      total += v.tensor.Data()[i];
    }
    double mean = total / static_cast<double>(v.tensor.size);
    return Value::F64(mean);
  }
  if (name == "len") {
    expect_args(1, name);
    const Value& v = args[0];
    if (v.type == DType::kTensor) {
      return Value::I64(v.tensor.size);
    }
    if (v.type == DType::kTuple) {
      return Value::I64(static_cast<int64_t>(v.tuple.elements.size()));
    }
    if (v.type == DType::kRecord) {
      return Value::I64(static_cast<int64_t>(v.record.fields.size()));
    }
    throw util::Error("len expects tuple, record, or tensor", 0, 0);
  }
  if (name == "keys") {
    expect_args(1, name);
    const Value& v = args[0];
    if (v.type != DType::kRecord) {
      throw util::Error("keys expects a record", 0, 0);
    }
    std::vector<Value> elems;
    elems.reserve(v.record.fields.size());
    for (const auto& f : v.record.fields) {
      elems.push_back(Value::String(f.first));
    }
    return Value::Tuple(std::move(elems));
  }
  if (name == "values") {
    expect_args(1, name);
    const Value& v = args[0];
    if (v.type != DType::kRecord) {
      throw util::Error("values expects a record", 0, 0);
    }
    std::vector<Value> elems;
    elems.reserve(v.record.fields.size());
    for (const auto& f : v.record.fields) {
      elems.push_back(f.second);
    }
    return Value::Tuple(std::move(elems));
  }
  if (name == "has_key") {
    expect_args(2, name);
    const Value& v = args[0];
    const Value& key = args[1];
    if (v.type != DType::kRecord || key.type != DType::kString) {
      throw util::Error("has_key expects (record, string)", 0, 0);
    }
    return Value::Bool(v.record.index.find(key.str) != v.record.index.end());
  }
  if (name == "floor") {
    expect_args(1, name);
    ensure_not_tensor(args[0], name);
    if (IsFloat(args[0].type) || args[0].type == DType::kDecimal) {
      return Value::F64(std::floor(args[0].f64));
    }
    return args[0];
  }
  if (name == "ceil") {
    expect_args(1, name);
    ensure_not_tensor(args[0], name);
    if (IsFloat(args[0].type) || args[0].type == DType::kDecimal) {
      return Value::F64(std::ceil(args[0].f64));
    }
    return args[0];
  }
  if (name == "round") {
    expect_args(1, name);
    ensure_not_tensor(args[0], name);
    if (IsFloat(args[0].type) || args[0].type == DType::kDecimal) {
      return Value::F64(std::round(args[0].f64));
    }
    return args[0];
  }
  if (name == "clamp") {
    expect_args(3, name);
    ensure_not_tensor(args[0], name);
    ensure_not_tensor(args[1], name);
    ensure_not_tensor(args[2], name);
    DType target = PromoteType(PromoteType(args[0].type, args[1].type), args[2].type);
    Value v0 = CastTo(target, args[0]);
    Value v1 = CastTo(target, args[1]);
    Value v2 = CastTo(target, args[2]);
    if (IsFloat(target)) {
      double res = std::clamp(v0.f64, v1.f64, v2.f64);
      return target == DType::kF32 ? Value::F32(static_cast<float>(res)) : Value::F64(res);
    }
    int64_t res = std::clamp(v0.i64, v1.i64, v2.i64);
    return Value::I64(res);
  }
  if (name == "min") {
    expect_args(2, name);
    ensure_not_tensor(args[0], name);
    ensure_not_tensor(args[1], name);
    DType target = PromoteType(args[0].type, args[1].type);
    Value v0 = CastTo(target, args[0]);
    Value v1 = CastTo(target, args[1]);
    if (IsFloat(target)) {
      double res = std::min(v0.f64, v1.f64);
      return target == DType::kF32 ? Value::F32(static_cast<float>(res)) : Value::F64(res);
    }
    int64_t res = std::min(v0.i64, v1.i64);
    return Value::I64(res);
  }
  if (name == "max") {
    expect_args(2, name);
    ensure_not_tensor(args[0], name);
    ensure_not_tensor(args[1], name);
    DType target = PromoteType(args[0].type, args[1].type);
    Value v0 = CastTo(target, args[0]);
    Value v1 = CastTo(target, args[1]);
    if (IsFloat(target)) {
      double res = std::max(v0.f64, v1.f64);
      return target == DType::kF32 ? Value::F32(static_cast<float>(res)) : Value::F64(res);
    }
    int64_t res = std::max(v0.i64, v1.i64);
    return Value::I64(res);
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
  if (stmt.tuple_pattern) {
    Value rhs = Evaluate(*stmt.value);
    if (rhs.type != DType::kTuple) {
      throw util::Error("Destructuring expects a tuple value", stmt.tuple_pattern->line,
                        stmt.tuple_pattern->column);
    }
    if (rhs.tuple.elements.size() != stmt.tuple_pattern->names.size()) {
      throw util::Error("Tuple destructuring arity mismatch", stmt.tuple_pattern->line,
                        stmt.tuple_pattern->column);
    }
    for (size_t i = 0; i < stmt.tuple_pattern->names.size(); ++i) {
      env_->Define(stmt.tuple_pattern->names[i], rhs.tuple.elements[i]);
    }
    return ExecResult{rhs, ControlSignal::kNone};
  }
  if (stmt.record_pattern) {
    Value rhs = Evaluate(*stmt.value);
    if (rhs.type != DType::kRecord) {
      throw util::Error("Destructuring expects a record value", stmt.record_pattern->line,
                        stmt.record_pattern->column);
    }
    if (rhs.record.fields.size() != stmt.record_pattern->fields.size()) {
      throw util::Error("Record destructuring field mismatch", stmt.record_pattern->line,
                        stmt.record_pattern->column);
    }
    for (size_t i = 0; i < stmt.record_pattern->fields.size(); ++i) {
      const auto& key = stmt.record_pattern->fields[i].first;
      if (rhs.record.index.find(key) == rhs.record.index.end()) {
        throw util::Error("Record key not found in destructuring: " + key,
                          stmt.record_pattern->line, stmt.record_pattern->column);
      }
      size_t idx = rhs.record.index[key];
      env_->Define(stmt.record_pattern->fields[i].second, rhs.record.fields[idx].second);
    }
    return ExecResult{rhs, ControlSignal::kNone};
  }
  Value value = Evaluate(*stmt.value);
  if (stmt.annotation.type) {
    if (!ValueMatchesType(value, stmt.annotation.type->name)) {
      throw util::Error("Type mismatch for variable '" + stmt.name + "'", stmt.line, stmt.column);
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
    case DType::kString:
      return "\"" + str + "\"";
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
    case DType::kTensor: {
      std::ostringstream oss;
      oss << "tensor" << ShapeToString(tensor.shape) << "<" << DTypeToString(tensor.elem_type)
          << ">";
      return oss.str();
    }
    case DType::kTuple: {
      std::ostringstream oss;
      oss << "(";
      for (size_t i = 0; i < tuple.elements.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << tuple.elements[i].ToString();
      }
      if (tuple.elements.size() == 1) oss << ",";  // singleton tuple syntax
      oss << ")";
      return oss.str();
    }
    case DType::kRecord: {
      std::ostringstream oss;
      oss << "{";
      for (size_t i = 0; i < record.fields.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << record.fields[i].first << ": " << record.fields[i].second.ToString();
      }
      oss << "}";
      return oss.str();
    }
    case DType::kFunction:
      return "<function>";
  }
  return "<unknown>";
}

}  // namespace lattice::runtime
