#include "runtime/ops.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

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

bool IsDenseTensor(const Value& v) {
  return v.type == DType::kTensor && v.tensor.kind == TensorKind::kDense;
}

bool RequireBool(const Value& v, int line, int column) {
  if (v.type != DType::kBool) {
    throw util::Error("Condition must be bool", line, column);
  }
  return v.boolean;
}

DType PromoteTensorElem(DType a, DType b, int line, int column) {
  return PromoteType(a, b, line, column);
}

// Forward declarations for helpers.
Value ToDenseTensor(const Value& v, int line, int column);

// Philox2x32-10 and Threefry2x32 implementations for deterministic RNG.
uint64_t MulHi32(uint32_t a, uint32_t b) {
  uint64_t res = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
  return res >> 32;
}

uint64_t Philox2x32(uint64_t counter, uint64_t key) {
  uint32_t c0 = static_cast<uint32_t>(counter);
  uint32_t c1 = static_cast<uint32_t>(counter >> 32);
  uint32_t k0 = static_cast<uint32_t>(key);
  uint32_t k1 = static_cast<uint32_t>(key >> 32);
  constexpr uint32_t kPhiloxW = 0x9E3779B9;
  constexpr uint32_t kPhiloxM0 = 0xD256D193;
  for (int round = 0; round < 10; ++round) {
    uint64_t hi = MulHi32(c0, kPhiloxM0);
    uint64_t lo = static_cast<uint64_t>(c0) * kPhiloxM0;
    uint32_t new_c0 = static_cast<uint32_t>(hi) ^ k0 ^ c1;
    uint32_t new_c1 = static_cast<uint32_t>(lo);
    c0 = new_c0;
    c1 = new_c1;
    k0 += kPhiloxW;
    k1 += kPhiloxW;
  }
  return (static_cast<uint64_t>(c1) << 32) | c0;
}

uint64_t Threefry2x32(uint64_t counter, uint64_t key) {
  uint32_t x0 = static_cast<uint32_t>(counter);
  uint32_t x1 = static_cast<uint32_t>(counter >> 32);
  uint32_t k0 = static_cast<uint32_t>(key);
  uint32_t k1 = static_cast<uint32_t>(key >> 32);
  constexpr uint32_t kConst = 0x1BD11BDA;
  uint32_t k2 = kConst ^ k0 ^ k1;
  auto rotl = [](uint32_t v, uint32_t r) { return (v << r) | (v >> (32 - r)); };
  auto inject_key = [&](int s) {
    switch (s % 3) {
      case 0:
        x0 += k0;
        x1 += k1;
        break;
      case 1:
        x0 += k1;
        x1 += k2;
        break;
      case 2:
        x0 += k2;
        x1 += k0;
        break;
    }
    x1 += static_cast<uint32_t>(s);
  };
  inject_key(0);
  const uint32_t R[8] = {13, 15, 26, 6, 17, 29, 16, 24};
  for (int round = 0; round < 8; ++round) {
    x0 += x1;
    x1 = rotl(x1, R[round]);
    x1 ^= x0;
    if ((round & 1) == 1) {
      inject_key((round + 1) / 2);
    }
  }
  return (static_cast<uint64_t>(x1) << 32) | x0;
}

double Uint64ToUnitDouble(uint64_t v) {
  // Use upper 53 bits for a double in [0,1).
  constexpr double kInv = 1.0 / static_cast<double>(uint64_t{1} << 53);
  return (static_cast<double>(v >> 11)) * kInv;
}

double ScalarF64(const Value& v, int line, int column) {
  if (v.type == DType::kTensor) {
    Value dense = ToDenseTensor(v, line, column);
    if (dense.tensor.size != 1) {
      throw util::Error("Expected scalar tensor", line, column);
    }
    return dense.tensor.Data()[0];
  }
  if (v.type == DType::kTensor) {
    throw util::Error("Expected scalar value", line, column);
  }
  return v.f64;
}

std::vector<double> Tensor1DToVector(const Value& v, int line, int column) {
  if (v.type != DType::kTensor) {
    throw util::Error("Expected a tensor argument", line, column);
  }
  Value dense = ToDenseTensor(v, line, column);
  if (dense.tensor.shape.size() != 1) {
    throw util::Error("Expected a 1D tensor", line, column);
  }
  std::vector<double> out(static_cast<size_t>(dense.tensor.size), 0.0);
  for (int64_t i = 0; i < dense.tensor.size; ++i)
    out[static_cast<size_t>(i)] = dense.tensor.Data()[i];
  return out;
}

Value ToDenseTensor(const Value& v, int line, int column) {
  if (v.type != DType::kTensor) return v;
  if (v.tensor.kind == TensorKind::kDense) return v;
  if (v.tensor.kind == TensorKind::kSparseCSR) {
    Value out = Value::Tensor(v.tensor.shape, v.tensor.elem_type, 0.0);
    for (size_t row = 0; row + 1 < v.tensor.indptr.size(); ++row) {
      int64_t start = v.tensor.indptr[row];
      int64_t end = v.tensor.indptr[row + 1];
      const int64_t row_i = static_cast<int64_t>(row);
      for (int64_t idx = start; idx < end; ++idx) {
        int64_t col = v.tensor.indices[idx];
        int64_t offset = row_i * v.tensor.shape[1] + col;
        out.tensor.Data()[offset] = v.tensor.sparse_values[idx];
      }
    }
    return out;
  }
  if (v.tensor.kind == TensorKind::kSparseCOO) {
    Value out = Value::Tensor(v.tensor.shape, v.tensor.elem_type, 0.0);
    for (size_t i = 0; i < v.tensor.rows.size(); ++i) {
      int64_t row = v.tensor.rows[i];
      int64_t col = v.tensor.cols[i];
      int64_t offset = row * v.tensor.shape[1] + col;
      out.tensor.Data()[offset] = v.tensor.sparse_values[i];
    }
    return out;
  }
  throw util::Error("Ragged tensor must be densified explicitly via to_dense", line, column);
}

Value DenseToCSR(const Value& dense) {
  int64_t rows = dense.tensor.shape[0];
  int64_t cols = dense.tensor.shape[1];
  std::vector<int64_t> indptr(rows + 1, 0);
  std::vector<int64_t> indices;
  std::vector<double> vals;
  const double* data = dense.tensor.Data();
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      double val = data[r * cols + c];
      if (val != 0.0) {
        indices.push_back(c);
        vals.push_back(val);
      }
    }
    indptr[static_cast<size_t>(r + 1)] = static_cast<int64_t>(indices.size());
  }
  return Value::TensorSparseCSR({rows, cols}, std::move(indptr), std::move(indices),
                                std::move(vals), dense.tensor.elem_type);
}

Value DenseToCOO(const Value& dense) {
  int64_t rows = dense.tensor.shape[0];
  int64_t cols = dense.tensor.shape[1];
  std::vector<int64_t> r;
  std::vector<int64_t> c;
  std::vector<double> vals;
  const double* data = dense.tensor.Data();
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      double val = data[i * cols + j];
      if (val != 0.0) {
        r.push_back(i);
        c.push_back(j);
        vals.push_back(val);
      }
    }
  }
  return Value::TensorSparseCOO({rows, cols}, std::move(r), std::move(c), std::move(vals),
                                dense.tensor.elem_type);
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

DType PromoteType(DType a, DType b, int line, int column) {
  auto is_decimal_or_rational = [](DType t) {
    return t == DType::kDecimal || t == DType::kRational;
  };
  if (a == DType::kString || b == DType::kString) {
    if (a == b) return DType::kString;
    throw util::Error("String can only operate with string", line, column);
  }
  if (a == DType::kTensor || b == DType::kTensor) {
    return DType::kTensor;
  }
  if (a == DType::kDecimal && b == DType::kDecimal) return DType::kDecimal;
  if (a == DType::kRational && b == DType::kRational) return DType::kRational;
  if (is_decimal_or_rational(a) || is_decimal_or_rational(b)) {
    throw util::Error("Decimal/rational cross-type arithmetic not supported in promotion", line,
                      column);
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

Value CastTo(DType target, const Value& v, int line, int column) {
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
        throw util::Error("Cannot cast complex/decimal/rational/string/function", line, column);
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
        throw util::Error("Cannot cast complex/decimal/rational/function", line, column);
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
        throw util::Error("Cannot cast complex/decimal/rational/function", line, column);
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
      throw util::Error("Cannot cast to string", line, column);
    case DType::kRational: {
      if (v.type == DType::kRational) {
        return v;
      }
      int64_t num = static_cast<int64_t>(as_signed());
      return Value::RationalValueNormalized(num, 1);
    }
    default:
      throw util::Error("Unsupported cast target", line, column);
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

Evaluator::Evaluator(std::shared_ptr<Environment> env) : env_(std::move(env)) {}

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
    // Handle ragged special-case.
    if ((lhs.type == DType::kTensor && lhs.tensor.kind == TensorKind::kRagged) ||
        (rhs.type == DType::kTensor && rhs.tensor.kind == TensorKind::kRagged)) {
      if (lhs.type != DType::kTensor || rhs.type != DType::kTensor ||
          lhs.tensor.kind != TensorKind::kRagged || rhs.tensor.kind != TensorKind::kRagged) {
        throw util::Error(
            "Ragged tensors only support ops with matching ragged tensors (use to_dense)",
            expr.line, expr.column);
      }
      if (lhs.tensor.row_splits != rhs.tensor.row_splits) {
        throw util::Error("Ragged tensors must share identical row_splits for elementwise ops",
                          expr.line, expr.column);
      }
      if (lhs.tensor.ragged_values.size() != rhs.tensor.ragged_values.size()) {
        throw util::Error("Ragged value buffers must match in length", expr.line, expr.column);
      }
      Value out = Value::TensorRagged(
          lhs.tensor.row_splits, {},
          PromoteType(lhs.tensor.elem_type, rhs.tensor.elem_type, expr.line, expr.column));
      out.tensor.ragged_values.resize(lhs.tensor.ragged_values.size());
      for (size_t i = 0; i < out.tensor.ragged_values.size(); ++i) {
        double lv = lhs.tensor.ragged_values[i];
        double rv = rhs.tensor.ragged_values[i];
        switch (expr.op) {
          case parser::BinaryOp::kAdd:
            out.tensor.ragged_values[i] = lv + rv;
            break;
          case parser::BinaryOp::kSub:
            out.tensor.ragged_values[i] = lv - rv;
            break;
          case parser::BinaryOp::kMul:
            out.tensor.ragged_values[i] = lv * rv;
            break;
          case parser::BinaryOp::kDiv:
            if (rv == 0.0) throw util::Error("Division by zero", expr.line, expr.column);
            out.tensor.ragged_values[i] = lv / rv;
            break;
          default:
            throw util::Error("Ragged tensors only support +,-,*,/", expr.line, expr.column);
        }
      }
      return out;
    }
    // Handle sparse: same format only, no broadcasting. Otherwise densify.
    if ((lhs.type == DType::kTensor && lhs.tensor.kind != TensorKind::kDense) ||
        (rhs.type == DType::kTensor && rhs.tensor.kind != TensorKind::kDense)) {
      if (lhs.type == DType::kTensor && rhs.type == DType::kTensor &&
          lhs.tensor.kind == rhs.tensor.kind &&
          (lhs.tensor.kind == TensorKind::kSparseCSR ||
           lhs.tensor.kind == TensorKind::kSparseCOO)) {
        if (lhs.tensor.shape != rhs.tensor.shape) {
          throw util::Error("Sparse tensors must share shape for elementwise ops", expr.line,
                            expr.column);
        }
        Value dense_l = ToDenseTensor(lhs, expr.line, expr.column);
        Value dense_r = ToDenseTensor(rhs, expr.line, expr.column);
        const std::vector<int64_t>& out_shape = lhs.tensor.shape;
        DType elem_target =
            PromoteType(dense_l.tensor.elem_type, dense_r.tensor.elem_type, expr.line, expr.column);
        Value out = Value::Tensor(out_shape, elem_target, 0.0);
        for (int64_t i = 0; i < out.tensor.size; ++i) {
          double lv = dense_l.tensor.Data()[i];
          double rv = dense_r.tensor.Data()[i];
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
              if (rv == 0.0) throw util::Error("Division by zero", expr.line, expr.column);
              out.tensor.Data()[i] = lv / rv;
              break;
            default:
              throw util::Error("Sparse tensor comparison not supported", expr.line, expr.column);
          }
        }
        return lhs.tensor.kind == TensorKind::kSparseCSR ? DenseToCSR(out) : DenseToCOO(out);
      }
      // Different sparse formats or sparse with ragged already handled above: error if both are
      // sparse but formats differ; else densify the sparse side when mixed with dense.
      if (lhs.type == DType::kTensor && rhs.type == DType::kTensor &&
          lhs.tensor.kind != rhs.tensor.kind && lhs.tensor.kind != TensorKind::kDense &&
          rhs.tensor.kind != TensorKind::kDense) {
        throw util::Error("Sparse tensor formats must match; convert first", expr.line,
                          expr.column);
      }
      Value lhs_d = lhs.type == DType::kTensor && lhs.tensor.kind != TensorKind::kDense
                        ? ToDenseTensor(lhs, expr.line, expr.column)
                        : lhs;
      Value rhs_d = rhs.type == DType::kTensor && rhs.tensor.kind != TensorKind::kDense
                        ? ToDenseTensor(rhs, expr.line, expr.column)
                        : rhs;
      lhs = lhs_d;
      rhs = rhs_d;
    }
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
      elem_target = PromoteType(lhs.tensor.elem_type, rhs.tensor.elem_type, expr.line, expr.column);
    } else if (lhs.type == DType::kTensor) {
      elem_target = PromoteType(lhs.tensor.elem_type, rhs.type, expr.line, expr.column);
    } else {
      elem_target = PromoteType(rhs.tensor.elem_type, lhs.type, expr.line, expr.column);
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

  DType target = PromoteType(lhs.type, rhs.type, expr.line, expr.column);
  Value lcast = CastTo(target, lhs, expr.line, expr.column);
  Value rcast = CastTo(target, rhs, expr.line, expr.column);

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
  if (!env_) {
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
  if (!env_) {
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
    return CastTo(dt.value(), v, call.line, call.column);
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
    auto fn_env = std::make_shared<Environment>(fn->defining_env);
    for (size_t i = 0; i < args.size(); ++i) {
      if (!fn->parameter_types.empty() && i < fn->parameter_types.size()) {
        const std::string& annot = fn->parameter_types[i];
        if (!annot.empty() && !ValueMatchesType(args[i], annot)) {
          throw util::Error("Type mismatch for parameter '" + fn->parameters[i] + "' (expected " +
                                annot + ", got " + ValueTypeName(args[i]) + ")",
                            call.line, call.column);
        }
      }
      fn_env->Define(fn->parameters[i], args[i]);
    }
    Evaluator fn_evaluator(fn_env);
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
  int call_line = call.line;
  int call_col = call.column;
  auto expect_args = [&](size_t count, const std::string& func) {
    if (args.size() != count) {
      throw util::Error(func + " expects " + std::to_string(count) + " arguments", call_line,
                        call_col);
    }
  };
  auto ensure_not_tensor = [&](const Value& v, const std::string& func) {
    if (v.type == DType::kTensor) {
      throw util::Error(func + " does not support tensor arguments", call_line, call_col);
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
    return CastTo(DType::kI64, args[0], call_line, call_col);
  }
  if (name == "float") {
    expect_args(1, name);
    return CastTo(DType::kF64, args[0], call_line, call_col);
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
      throw util::Error("rational denominator cannot be zero", call_line, call_col);
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
      throw util::Error("tensor expects at least shape and fill value", call_line, call_col);
    }
    std::vector<int64_t> shape;
    shape.reserve(args.size() - 1);
    for (size_t i = 0; i + 1 < args.size(); ++i) {
      shape.push_back(static_cast<int64_t>(args[i].f64));
    }
    double fill = args.back().f64;
    if (shape.empty()) {
      throw util::Error("tensor shape cannot be empty", call_line, call_col);
    }
    for (int64_t dim : shape) {
      if (dim <= 0) {
        throw util::Error("tensor dimensions must be positive (got " + std::to_string(dim) + ")",
                          call_line, call_col);
      }
    }
    DType elem_type = DType::kF64;
    return Value::Tensor(shape, elem_type, fill);
  }
  if (name == "tensor_values") {
    if (args.empty()) {
      throw util::Error("tensor_values expects at least one value", call_line, call_col);
    }
    for (const auto& v : args) {
      if (v.type == DType::kTensor) {
        throw util::Error("tensor_values does not accept tensor arguments", call_line, call_col);
      }
    }
    // Support nested tuple/list literals to build multi-dimensional tensors.
    struct FlattenState {
      std::vector<int64_t> shape;
      std::vector<Value> leaves;
      DType elem_type;
      bool has_elem = false;
    } st;
    std::function<void(const Value&, size_t)> flatten = [&](const Value& v, size_t depth) {
      if (v.type == DType::kTuple) {
        int64_t len = static_cast<int64_t>(v.tuple.elements.size());
        if (depth == st.shape.size()) {
          st.shape.push_back(len);
        } else if (st.shape[depth] != len) {
          throw util::Error("tensor_values nested lists must have consistent lengths", call_line,
                            call_col);
        }
        for (const auto& e : v.tuple.elements) {
          flatten(e, depth + 1);
        }
        return;
      }
      st.leaves.push_back(v);
      if (!st.has_elem) {
        st.elem_type = v.type;
        st.has_elem = true;
      } else {
        st.elem_type = PromoteType(st.elem_type, v.type, call_line, call_col);
      }
    };

    if (args.size() == 1 && args[0].type == DType::kTuple) {
      flatten(args[0], 0);
    } else {
      // Original flat 1-D behavior.
      st.shape.push_back(static_cast<int64_t>(args.size()));
      for (const auto& v : args) flatten(v, 1);  // depth offset to keep shape size=1
    }

    if (!st.has_elem || st.leaves.empty()) {
      throw util::Error("tensor_values needs at least one scalar", call_line, call_col);
    }
    int64_t total = 1;
    for (int64_t d : st.shape) {
      if (d <= 0) {
        throw util::Error("tensor dimensions must be positive", call_line, call_col);
      }
      total *= d;
    }
    if (static_cast<int64_t>(st.leaves.size()) != total) {
      throw util::Error("tensor_values data does not match inferred shape", call_line, call_col);
    }
    Value out = Value::Tensor(st.shape, st.elem_type, 0.0);
    for (size_t i = 0; i < st.leaves.size(); ++i) {
      out.tensor.Data()[static_cast<int64_t>(i)] =
          CastTo(st.elem_type, st.leaves[i], call_line, call_col).f64;
    }
    out.tensor.elem_type = st.elem_type;
    return out;
  }
  if (name == "tensor_sparse_csr") {
    if (args.size() != 4) {
      throw util::Error(
          "tensor_sparse_csr expects (shape_tuple, indptr_tuple, indices_tuple, values_tuple)",
          call_line, call_col);
    }
    auto extract_tuple_i64 = [&](const Value& v, const std::string& what) {
      if (v.type != DType::kTuple) {
        throw util::Error(what + " must be a tuple", call_line, call_col);
      }
      std::vector<int64_t> out;
      out.reserve(v.tuple.elements.size());
      for (const auto& e : v.tuple.elements) {
        out.push_back(static_cast<int64_t>(e.f64));
      }
      return out;
    };
    auto shape = extract_tuple_i64(args[0], "shape");
    if (shape.size() != 2) {
      throw util::Error("tensor_sparse_csr shape must be length 2", call_line, call_col);
    }
    auto indptr = extract_tuple_i64(args[1], "indptr");
    auto indices = extract_tuple_i64(args[2], "indices");
    if (indptr.size() != static_cast<size_t>(shape[0] + 1)) {
      throw util::Error("indptr length must be rows+1", call_line, call_col);
    }
    auto vals_tuple = args[3];
    if (vals_tuple.type != DType::kTuple) {
      throw util::Error("values must be a tuple", call_line, call_col);
    }
    if (indices.size() != vals_tuple.tuple.elements.size()) {
      throw util::Error("indices and values must have same length", call_line, call_col);
    }
    std::vector<double> vals;
    vals.reserve(vals_tuple.tuple.elements.size());
    DType elem_type = DType::kF64;
    for (const auto& e : vals_tuple.tuple.elements) {
      vals.push_back(e.f64);
      elem_type = PromoteType(elem_type, e.type, call_line, call_col);
    }
    return Value::TensorSparseCSR(std::move(shape), std::move(indptr), std::move(indices),
                                  std::move(vals), elem_type);
  }
  if (name == "tensor_sparse_coo") {
    if (args.size() != 4) {
      throw util::Error(
          "tensor_sparse_coo expects (shape_tuple, rows_tuple, cols_tuple, values_tuple)",
          call_line, call_col);
    }
    auto extract_tuple_i64 = [&](const Value& v, const std::string& what) {
      if (v.type != DType::kTuple) {
        throw util::Error(what + " must be a tuple", call_line, call_col);
      }
      std::vector<int64_t> out;
      out.reserve(v.tuple.elements.size());
      for (const auto& e : v.tuple.elements) {
        out.push_back(static_cast<int64_t>(e.f64));
      }
      return out;
    };
    auto shape = extract_tuple_i64(args[0], "shape");
    if (shape.size() != 2) {
      throw util::Error("tensor_sparse_coo shape must be length 2", call_line, call_col);
    }
    auto rows_v = extract_tuple_i64(args[1], "rows");
    auto cols_v = extract_tuple_i64(args[2], "cols");
    auto vals_tuple = args[3];
    if (vals_tuple.type != DType::kTuple) {
      throw util::Error("values must be a tuple", call_line, call_col);
    }
    if (rows_v.size() != cols_v.size() || rows_v.size() != vals_tuple.tuple.elements.size()) {
      throw util::Error("rows, cols, and values must have same length", call_line, call_col);
    }
    std::vector<double> vals;
    vals.reserve(vals_tuple.tuple.elements.size());
    DType elem_type = DType::kF64;
    for (const auto& e : vals_tuple.tuple.elements) {
      vals.push_back(e.f64);
      elem_type = PromoteType(elem_type, e.type, call_line, call_col);
    }
    return Value::TensorSparseCOO(std::move(shape), std::move(rows_v), std::move(cols_v),
                                  std::move(vals), elem_type);
  }
  if (name == "tensor_ragged") {
    if (args.size() != 2) {
      throw util::Error("tensor_ragged expects (row_splits_tuple, values_tuple)", call_line,
                        call_col);
    }
    auto extract_tuple_i64 = [&](const Value& v, const std::string& what) {
      if (v.type != DType::kTuple) {
        throw util::Error(what + " must be a tuple", call_line, call_col);
      }
      std::vector<int64_t> out;
      out.reserve(v.tuple.elements.size());
      for (const auto& e : v.tuple.elements) {
        out.push_back(static_cast<int64_t>(e.f64));
      }
      return out;
    };
    auto splits = extract_tuple_i64(args[0], "row_splits");
    if (splits.size() < 2) {
      throw util::Error("row_splits must have at least 2 entries", call_line, call_col);
    }
    for (size_t i = 1; i < splits.size(); ++i) {
      if (splits[i] < splits[i - 1]) {
        throw util::Error("row_splits must be non-decreasing", call_line, call_col);
      }
    }
    auto vals_tuple = args[1];
    if (vals_tuple.type != DType::kTuple) {
      throw util::Error("values must be a tuple", call_line, call_col);
    }
    std::vector<double> vals;
    vals.reserve(vals_tuple.tuple.elements.size());
    DType elem_type = DType::kF64;
    for (const auto& e : vals_tuple.tuple.elements) {
      vals.push_back(e.f64);
      elem_type = PromoteType(elem_type, e.type, call_line, call_col);
    }
    return Value::TensorRagged(std::move(splits), std::move(vals), elem_type);
  }
  if (name == "to_dense") {
    expect_args(1, name);
    const Value& v = args[0];
    if (v.type != DType::kTensor) {
      throw util::Error("to_dense expects a tensor", call_line, call_col);
    }
    if (v.tensor.kind == TensorKind::kDense) return v;
    // Only support 2D sparse -> dense; ragged not supported.
    if (v.tensor.kind == TensorKind::kSparseCSR) {
      Value out = Value::Tensor(v.tensor.shape, v.tensor.elem_type, 0.0);
      for (size_t row = 0; row + 1 < v.tensor.indptr.size(); ++row) {
        int64_t start = v.tensor.indptr[row];
        int64_t end = v.tensor.indptr[row + 1];
        const int64_t row_i = static_cast<int64_t>(row);
        for (int64_t idx = start; idx < end; ++idx) {
          int64_t col = v.tensor.indices[idx];
          int64_t offset = row_i * v.tensor.shape[1] + col;
          out.tensor.Data()[offset] = v.tensor.sparse_values[idx];
        }
      }
      return out;
    }
    if (v.tensor.kind == TensorKind::kSparseCOO) {
      Value out = Value::Tensor(v.tensor.shape, v.tensor.elem_type, 0.0);
      for (size_t i = 0; i < v.tensor.rows.size(); ++i) {
        int64_t row = v.tensor.rows[i];
        int64_t col = v.tensor.cols[i];
        int64_t offset = row * v.tensor.shape[1] + col;
        out.tensor.Data()[offset] = v.tensor.sparse_values[i];
      }
      return out;
    }
    throw util::Error("to_dense for ragged tensors is not supported yet", call_line, call_col);
  }
  if (name == "to_sparse_csr") {
    expect_args(1, name);
    const Value& v = args[0];
    if (!IsDenseTensor(v)) {
      throw util::Error("to_sparse_csr expects a dense tensor", call_line, call_col);
    }
    if (v.tensor.shape.size() != 2) {
      throw util::Error("to_sparse_csr supports only 2D tensors", call_line, call_col);
    }
    int64_t rows = v.tensor.shape[0];
    int64_t cols = v.tensor.shape[1];
    std::vector<int64_t> indptr(rows + 1, 0);
    std::vector<int64_t> indices;
    std::vector<double> vals;
    const double* data = v.tensor.Data();
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        double val = data[r * cols + c];
        if (val != 0.0) {
          indices.push_back(c);
          vals.push_back(val);
        }
      }
      indptr[static_cast<size_t>(r + 1)] = static_cast<int64_t>(indices.size());
    }
    return Value::TensorSparseCSR({rows, cols}, std::move(indptr), std::move(indices),
                                  std::move(vals), v.tensor.elem_type);
  }
  if (name == "to_sparse_coo") {
    expect_args(1, name);
    const Value& v = args[0];
    if (!IsDenseTensor(v)) {
      throw util::Error("to_sparse_coo expects a dense tensor", call_line, call_col);
    }
    if (v.tensor.shape.size() != 2) {
      throw util::Error("to_sparse_coo supports only 2D tensors", call_line, call_col);
    }
    int64_t rows = v.tensor.shape[0];
    int64_t cols = v.tensor.shape[1];
    std::vector<int64_t> r;
    std::vector<int64_t> c;
    std::vector<double> vals;
    const double* data = v.tensor.Data();
    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        double val = data[i * cols + j];
        if (val != 0.0) {
          r.push_back(i);
          c.push_back(j);
          vals.push_back(val);
        }
      }
    }
    return Value::TensorSparseCOO({rows, cols}, std::move(r), std::move(c), std::move(vals),
                                  v.tensor.elem_type);
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
      throw util::Error("gcd requires integer arguments", call_line, call_col);
    }
    auto a = static_cast<long long>(CastTo(DType::kI64, args[0], call_line, call_col).i64);
    auto b = static_cast<long long>(CastTo(DType::kI64, args[1], call_line, call_col).i64);
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
      throw util::Error("lcm requires integer arguments", call_line, call_col);
    }
    auto a = static_cast<long long>(CastTo(DType::kI64, args[0], call_line, call_col).i64);
    auto b = static_cast<long long>(CastTo(DType::kI64, args[1], call_line, call_col).i64);
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
      throw util::Error("mod divisor cannot be zero", call_line, call_col);
    }
    if (!IsFloat(args[0].type) && !IsFloat(args[1].type) && !IsComplex(args[0].type) &&
        !IsComplex(args[1].type) && args[0].type != DType::kDecimal &&
        args[0].type != DType::kRational && args[1].type != DType::kDecimal &&
        args[1].type != DType::kRational) {
      auto lhs = CastTo(DType::kI64, args[0], call_line, call_col).i64;
      auto rhs = CastTo(DType::kI64, args[1], call_line, call_col).i64;
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
    auto finish_as_elem = [&](double total) {
      return CastTo(v.tensor.elem_type, Value::F64(total), call_line, call_col);
    };
    if (v.tensor.kind == TensorKind::kDense) {
      double total = 0.0;
      for (int64_t i = 0; i < v.tensor.size; ++i) {
        total += v.tensor.Data()[i];
      }
      return finish_as_elem(total);
    }
    if (v.tensor.kind == TensorKind::kSparseCSR || v.tensor.kind == TensorKind::kSparseCOO) {
      double total = 0.0;
      for (double val : v.tensor.sparse_values) total += val;
      return finish_as_elem(total);
    }
    if (v.tensor.kind == TensorKind::kRagged) {
      double total = 0.0;
      for (double val : v.tensor.ragged_values) total += val;
      return finish_as_elem(total);
    }
    throw util::Error("Unsupported tensor kind for sum", call_line, call_col);
  }
  if (name == "mean") {
    expect_args(1, name);
    const Value& v = args[0];
    if (v.type != DType::kTensor) {
      return Value::F64(v.f64);
    }
    auto finish_as_elem = [&](double mean) {
      return CastTo(v.tensor.elem_type, Value::F64(mean), call_line, call_col);
    };
    if (v.tensor.kind == TensorKind::kDense) {
      double total = 0.0;
      for (int64_t i = 0; i < v.tensor.size; ++i) {
        total += v.tensor.Data()[i];
      }
      double mean = total / static_cast<double>(v.tensor.size);
      return finish_as_elem(mean);
    }
    if (v.tensor.kind == TensorKind::kSparseCSR || v.tensor.kind == TensorKind::kSparseCOO) {
      double total = 0.0;
      for (double val : v.tensor.sparse_values) total += val;
      double denom = static_cast<double>(v.tensor.shape[0] * v.tensor.shape[1]);
      return finish_as_elem(total / denom);
    }
    if (v.tensor.kind == TensorKind::kRagged) {
      double total = 0.0;
      for (double val : v.tensor.ragged_values) total += val;
      double denom = static_cast<double>(v.tensor.row_splits.back());
      return finish_as_elem(total / denom);
    }
    throw util::Error("Unsupported tensor kind for mean", call_line, call_col);
  }
  if (name == "var" || name == "std") {
    expect_args(1, name);
    const Value& v = args[0];
    if (v.type != DType::kTensor) {
      return Value::F64(0.0);
    }
    auto finish_as_elem = [&](double x) {
      return CastTo(v.tensor.elem_type, Value::F64(x), call_line, call_col);
    };
    auto compute = [&](auto accessor, double count) -> Value {
      double total = 0.0;
      for (double val : accessor()) total += val;
      double mean = total / count;
      double sum_sq = 0.0;
      for (double val : accessor()) {
        double diff = val - mean;
        sum_sq += diff * diff;
      }
      double variance = sum_sq / count;
      if (name == "var") return finish_as_elem(variance);
      return finish_as_elem(std::sqrt(variance));
    };
    if (v.tensor.kind == TensorKind::kDense) {
      return compute(
          [&]() { return std::vector<double>(v.tensor.Data(), v.tensor.Data() + v.tensor.size); },
          static_cast<double>(v.tensor.size));
    }
    if (v.tensor.kind == TensorKind::kSparseCSR || v.tensor.kind == TensorKind::kSparseCOO) {
      // Treat missing entries as zero.
      std::vector<double> dense_vals(static_cast<size_t>(v.tensor.shape[0] * v.tensor.shape[1]),
                                     0.0);
      if (v.tensor.kind == TensorKind::kSparseCSR) {
        for (size_t row = 0; row + 1 < v.tensor.indptr.size(); ++row) {
          int64_t start = v.tensor.indptr[row];
          int64_t end = v.tensor.indptr[row + 1];
          const int64_t row_i = static_cast<int64_t>(row);
          for (int64_t idx = start; idx < end; ++idx) {
            int64_t col = v.tensor.indices[idx];
            dense_vals[static_cast<size_t>(row_i * v.tensor.shape[1] + col)] =
                v.tensor.sparse_values[idx];
          }
        }
      } else {
        for (size_t i = 0; i < v.tensor.rows.size(); ++i) {
          int64_t row = v.tensor.rows[i];
          int64_t col = v.tensor.cols[i];
          dense_vals[static_cast<size_t>(row * v.tensor.shape[1] + col)] =
              v.tensor.sparse_values[i];
        }
      }
      double count = static_cast<double>(v.tensor.shape[0] * v.tensor.shape[1]);
      return compute([&]() { return dense_vals; }, count);
    }
    if (v.tensor.kind == TensorKind::kRagged) {
      double count = static_cast<double>(v.tensor.row_splits.back());
      return compute([&]() { return v.tensor.ragged_values; }, count);
    }
    throw util::Error("Unsupported tensor kind for variance/std", call_line, call_col);
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
    throw util::Error("len expects tuple, record, or tensor", call_line, call_col);
  }
  if (name == "transpose") {
    expect_args(1, name);
    const Value& v = args[0];
    if (v.type != DType::kTensor) {
      throw util::Error("transpose expects a tensor", call_line, call_col);
    }
    Value dense = ToDenseTensor(v, call_line, call_col);
    if (dense.tensor.shape.size() != 2) {
      throw util::Error("transpose supports only 2D tensors", call_line, call_col);
    }
    int64_t rows = dense.tensor.shape[0];
    int64_t cols = dense.tensor.shape[1];
    Value out = Value::Tensor(std::vector<int64_t>{cols, rows}, dense.tensor.elem_type, 0.0);
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        out.tensor.Data()[c * rows + r] = dense.tensor.Data()[r * cols + c];
      }
    }
    return out;
  }
  if (name == "matmul") {
    expect_args(2, name);
    Value lhs = args[0];
    Value rhs = args[1];
    if (lhs.type != DType::kTensor || rhs.type != DType::kTensor) {
      throw util::Error("matmul expects tensor arguments", call_line, call_col);
    }
    lhs = ToDenseTensor(lhs, call_line, call_col);
    rhs = ToDenseTensor(rhs, call_line, call_col);
    if (lhs.tensor.shape.size() != 2 || rhs.tensor.shape.size() != 2) {
      throw util::Error("matmul supports only 2D tensors", call_line, call_col);
    }
    int64_t m = lhs.tensor.shape[0];
    int64_t k = lhs.tensor.shape[1];
    int64_t k2 = rhs.tensor.shape[0];
    int64_t n = rhs.tensor.shape[1];
    if (k != k2) {
      throw util::Error("matmul shape mismatch", call_line, call_col);
    }
    DType elem = PromoteTensorElem(lhs.tensor.elem_type, rhs.tensor.elem_type, call_line, call_col);
    Value out = Value::Tensor(std::vector<int64_t>{m, n}, elem, 0.0);
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        double acc = 0.0;
        for (int64_t kk = 0; kk < k; ++kk) {
          acc += lhs.tensor.Data()[i * k + kk] * rhs.tensor.Data()[kk * n + j];
        }
        out.tensor.Data()[i * n + j] = acc;
      }
    }
    return out;
  }
  if (name == "conv2d") {
    expect_args(2, name);
    Value input = ToDenseTensor(args[0], call_line, call_col);
    Value kernel = ToDenseTensor(args[1], call_line, call_col);
    if (input.type != DType::kTensor || kernel.type != DType::kTensor) {
      throw util::Error("conv2d expects tensor arguments", call_line, call_col);
    }
    if (input.tensor.shape.size() != 2 || kernel.tensor.shape.size() != 2) {
      throw util::Error("conv2d supports 2D tensors only", call_line, call_col);
    }
    int64_t h = input.tensor.shape[0];
    int64_t w = input.tensor.shape[1];
    int64_t kh = kernel.tensor.shape[0];
    int64_t kw = kernel.tensor.shape[1];
    if (kh > h || kw > w) {
      throw util::Error("conv2d kernel larger than input", call_line, call_col);
    }
    int64_t oh = h - kh + 1;
    int64_t ow = w - kw + 1;
    DType elem =
        PromoteTensorElem(input.tensor.elem_type, kernel.tensor.elem_type, call_line, call_col);
    Value out = Value::Tensor({oh, ow}, elem, 0.0);
    for (int64_t i = 0; i < oh; ++i) {
      for (int64_t j = 0; j < ow; ++j) {
        double acc = 0.0;
        for (int64_t ki = 0; ki < kh; ++ki) {
          for (int64_t kj = 0; kj < kw; ++kj) {
            acc +=
                input.tensor.Data()[(i + ki) * w + (j + kj)] * kernel.tensor.Data()[ki * kw + kj];
          }
        }
        out.tensor.Data()[i * ow + j] = acc;
      }
    }
    return out;
  }
  if (name == "max_pool2d") {
    expect_args(3, name);
    Value input = ToDenseTensor(args[0], call_line, call_col);
    if (input.type != DType::kTensor || input.tensor.shape.size() != 2) {
      throw util::Error("max_pool2d expects a 2D tensor", call_line, call_col);
    }
    int64_t kh = CastTo(DType::kI64, args[1], call_line, call_col).i64;
    int64_t kw = CastTo(DType::kI64, args[2], call_line, call_col).i64;
    if (kh <= 0 || kw <= 0) {
      throw util::Error("max_pool2d kernel sizes must be positive", call_line, call_col);
    }
    int64_t h = input.tensor.shape[0];
    int64_t w = input.tensor.shape[1];
    int64_t oh = h / kh;
    int64_t ow = w / kw;
    Value out = Value::Tensor({oh, ow}, input.tensor.elem_type, 0.0);
    for (int64_t i = 0; i < oh; ++i) {
      for (int64_t j = 0; j < ow; ++j) {
        double m = -std::numeric_limits<double>::infinity();
        for (int64_t ki = 0; ki < kh; ++ki) {
          for (int64_t kj = 0; kj < kw; ++kj) {
            m = std::max(m, input.tensor.Data()[(i * kh + ki) * w + (j * kw + kj)]);
          }
        }
        out.tensor.Data()[i * ow + j] = m;
      }
    }
    return out;
  }
  if (name == "fft1d") {
    expect_args(1, name);
    Value input = ToDenseTensor(args[0], call_line, call_col);
    if (input.type != DType::kTensor || input.tensor.shape.size() != 1) {
      throw util::Error("fft1d expects a 1D tensor", call_line, call_col);
    }
    int64_t n = input.tensor.shape[0];
    std::vector<double> real(static_cast<size_t>(n), 0.0);
    std::vector<double> imag(static_cast<size_t>(n), 0.0);
    const double two_pi = 2.0 * M_PI;
    for (int64_t k = 0; k < n; ++k) {
      double sum_r = 0.0;
      double sum_i = 0.0;
      for (int64_t t = 0; t < n; ++t) {
        double angle = -two_pi * static_cast<double>(k * t) / static_cast<double>(n);
        double val = input.tensor.Data()[t];
        sum_r += val * std::cos(angle);
        sum_i += val * std::sin(angle);
      }
      real[static_cast<size_t>(k)] = sum_r;
      imag[static_cast<size_t>(k)] = sum_i;
    }
    Value real_t = Value::Tensor({n}, DType::kF64, 0.0);
    Value imag_t = Value::Tensor({n}, DType::kF64, 0.0);
    for (int64_t i = 0; i < n; ++i) {
      real_t.tensor.Data()[i] = real[static_cast<size_t>(i)];
      imag_t.tensor.Data()[i] = imag[static_cast<size_t>(i)];
    }
    return Value::Tuple({real_t, imag_t});
  }
  if (name == "solve") {
    expect_args(2, name);
    Value A = ToDenseTensor(args[0], call_line, call_col);
    Value B = ToDenseTensor(args[1], call_line, call_col);
    if (A.type != DType::kTensor || B.type != DType::kTensor) {
      throw util::Error("solve expects tensor arguments", call_line, call_col);
    }
    if (A.tensor.shape.size() != 2) {
      throw util::Error("solve expects a 2D coefficient matrix", call_line, call_col);
    }
    int64_t n = A.tensor.shape[0];
    if (A.tensor.shape[1] != n) {
      throw util::Error("solve requires a square matrix", call_line, call_col);
    }
    if (B.tensor.shape.size() == 1) {
      if (B.tensor.shape[0] != n) {
        throw util::Error("solve rhs length must match matrix rows", call_line, call_col);
      }
    } else if (B.tensor.shape.size() == 2) {
      if (B.tensor.shape[0] != n) {
        throw util::Error("solve rhs rows must match matrix rows", call_line, call_col);
      }
    } else {
      throw util::Error("solve rhs must be 1D or 2D tensor", call_line, call_col);
    }
    DType elem = PromoteTensorElem(A.tensor.elem_type, B.tensor.elem_type, call_line, call_col);
    std::vector<std::vector<double>> a(n, std::vector<double>(n));
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        a[static_cast<size_t>(i)][static_cast<size_t>(j)] =
            CastTo(DType::kF64, Value::F64(A.tensor.Data()[i * A.tensor.shape[1] + j]), call_line,
                   call_col)
                .f64;
      }
    }
    int64_t rhs_cols = B.tensor.shape.size() == 1 ? 1 : B.tensor.shape[1];
    std::vector<std::vector<double>> b(n, std::vector<double>(rhs_cols, 0.0));
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < rhs_cols; ++j) {
        int64_t idx = B.tensor.shape.size() == 1 ? i : (i * rhs_cols + j);
        b[static_cast<size_t>(i)][static_cast<size_t>(j)] =
            CastTo(DType::kF64, Value::F64(B.tensor.Data()[idx]), call_line, call_col).f64;
      }
    }
    for (int64_t k = 0; k < n; ++k) {
      int64_t pivot = k;
      double maxv = std::abs(a[static_cast<size_t>(k)][static_cast<size_t>(k)]);
      for (int64_t i = k + 1; i < n; ++i) {
        double v = std::abs(a[static_cast<size_t>(i)][static_cast<size_t>(k)]);
        if (v > maxv) {
          maxv = v;
          pivot = i;
        }
      }
      if (maxv == 0.0) {
        throw util::Error("solve singular matrix", call_line, call_col);
      }
      if (pivot != k) {
        std::swap(a[static_cast<size_t>(pivot)], a[static_cast<size_t>(k)]);
        std::swap(b[static_cast<size_t>(pivot)], b[static_cast<size_t>(k)]);
      }
      double diag = a[static_cast<size_t>(k)][static_cast<size_t>(k)];
      for (int64_t j = k; j < n; ++j) {
        a[static_cast<size_t>(k)][static_cast<size_t>(j)] /= diag;
      }
      for (int64_t j = 0; j < rhs_cols; ++j)
        b[static_cast<size_t>(k)][static_cast<size_t>(j)] /= diag;
      for (int64_t i = 0; i < n; ++i) {
        if (i == k) continue;
        double factor = a[static_cast<size_t>(i)][static_cast<size_t>(k)];
        for (int64_t j = k; j < n; ++j) {
          a[static_cast<size_t>(i)][static_cast<size_t>(j)] -=
              factor * a[static_cast<size_t>(k)][static_cast<size_t>(j)];
        }
        for (int64_t j = 0; j < rhs_cols; ++j) {
          b[static_cast<size_t>(i)][static_cast<size_t>(j)] -=
              factor * b[static_cast<size_t>(k)][static_cast<size_t>(j)];
        }
      }
    }
    std::vector<double> flat(static_cast<size_t>(n * rhs_cols), 0.0);
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < rhs_cols; ++j) {
        flat[static_cast<size_t>(i * rhs_cols + j)] =
            b[static_cast<size_t>(i)][static_cast<size_t>(j)];
      }
    }
    Value out = Value::Tensor({n, rhs_cols}, elem, 0.0);
    for (int64_t i = 0; i < n * rhs_cols; ++i) out.tensor.Data()[i] = flat[static_cast<size_t>(i)];
    if (rhs_cols == 1) {
      out.tensor.shape = {n};
      out.tensor.strides = {1};
      out.tensor.size = n;
    }
    return out;
  }
  auto make_dense_tensor = [&](const std::vector<double>& data, const std::vector<int64_t>& shape,
                               DType elem) {
    Value t = Value::Tensor(shape, elem, 0.0);
    for (size_t i = 0; i < data.size(); ++i) t.tensor.Data()[static_cast<int64_t>(i)] = data[i];
    return t;
  };
  if (name == "lu") {
    expect_args(1, name);
    Value A = ToDenseTensor(args[0], call_line, call_col);
    if (A.type != DType::kTensor || A.tensor.shape.size() != 2 ||
        A.tensor.shape[0] != A.tensor.shape[1]) {
      throw util::Error("lu expects a square 2D tensor", call_line, call_col);
    }
    int64_t n = A.tensor.shape[0];
    std::vector<double> L(static_cast<size_t>(n * n), 0.0);
    std::vector<double> U(static_cast<size_t>(n * n), 0.0);
    for (int64_t i = 0; i < n; ++i) L[static_cast<size_t>(i * n + i)] = 1.0;
    for (int64_t k = 0; k < n; ++k) {
      for (int64_t j = k; j < n; ++j) {
        double sum = 0.0;
        for (int64_t s = 0; s < k; ++s)
          sum += L[static_cast<size_t>(k * n + s)] * U[static_cast<size_t>(s * n + j)];
        U[static_cast<size_t>(k * n + j)] = A.tensor.Data()[k * A.tensor.shape[1] + j] - sum;
      }
      if (U[static_cast<size_t>(k * n + k)] == 0.0) {
        throw util::Error("lu singular matrix", call_line, call_col);
      }
      for (int64_t i = k + 1; i < n; ++i) {
        double sum = 0.0;
        for (int64_t s = 0; s < k; ++s)
          sum += L[static_cast<size_t>(i * n + s)] * U[static_cast<size_t>(s * n + k)];
        L[static_cast<size_t>(i * n + k)] =
            (A.tensor.Data()[i * A.tensor.shape[1] + k] - sum) / U[static_cast<size_t>(k * n + k)];
      }
    }
    Value l_t = make_dense_tensor(L, {n, n}, A.tensor.elem_type);
    Value u_t = make_dense_tensor(U, {n, n}, A.tensor.elem_type);
    return Value::Tuple({l_t, u_t});
  }
  if (name == "qr") {
    expect_args(1, name);
    Value A = ToDenseTensor(args[0], call_line, call_col);
    if (A.type != DType::kTensor || A.tensor.shape.size() != 2) {
      throw util::Error("qr expects a 2D tensor", call_line, call_col);
    }
    int64_t m = A.tensor.shape[0];
    int64_t n = A.tensor.shape[1];
    std::vector<std::vector<double>> q(m, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> r(n, std::vector<double>(n, 0.0));
    auto a_val = [&](int64_t i, int64_t j) { return A.tensor.Data()[i * n + j]; };
    for (int64_t j = 0; j < n; ++j) {
      for (int64_t i = 0; i < m; ++i)
        q[static_cast<size_t>(i)][static_cast<size_t>(j)] = a_val(i, j);
      for (int64_t k = 0; k < j; ++k) {
        double dot = 0.0;
        for (int64_t i = 0; i < m; ++i)
          dot += q[static_cast<size_t>(i)][static_cast<size_t>(j)] *
                 q[static_cast<size_t>(i)][static_cast<size_t>(k)];
        r[static_cast<size_t>(k)][static_cast<size_t>(j)] = dot;
        for (int64_t i = 0; i < m; ++i)
          q[static_cast<size_t>(i)][static_cast<size_t>(j)] -=
              dot * q[static_cast<size_t>(i)][static_cast<size_t>(k)];
      }
      double norm = 0.0;
      for (int64_t i = 0; i < m; ++i)
        norm += q[static_cast<size_t>(i)][static_cast<size_t>(j)] *
                q[static_cast<size_t>(i)][static_cast<size_t>(j)];
      norm = std::sqrt(norm);
      if (norm == 0.0) throw util::Error("qr rank deficiency", call_line, call_col);
      r[static_cast<size_t>(j)][static_cast<size_t>(j)] = norm;
      for (int64_t i = 0; i < m; ++i) q[static_cast<size_t>(i)][static_cast<size_t>(j)] /= norm;
    }
    std::vector<double> q_flat(static_cast<size_t>(m * n), 0.0);
    std::vector<double> r_flat(static_cast<size_t>(n * n), 0.0);
    for (int64_t i = 0; i < m; ++i)
      for (int64_t j = 0; j < n; ++j)
        q_flat[static_cast<size_t>(i * n + j)] = q[static_cast<size_t>(i)][static_cast<size_t>(j)];
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < n; ++j)
        r_flat[static_cast<size_t>(i * n + j)] = r[static_cast<size_t>(i)][static_cast<size_t>(j)];
    Value q_t = make_dense_tensor(q_flat, {m, n}, A.tensor.elem_type);
    Value r_t = make_dense_tensor(r_flat, {n, n}, A.tensor.elem_type);
    return Value::Tuple({q_t, r_t});
  }
  if (name == "svd") {
    expect_args(1, name);
    Value A = ToDenseTensor(args[0], call_line, call_col);
    if (A.type != DType::kTensor || A.tensor.shape.size() != 2) {
      throw util::Error("svd expects a 2D tensor", call_line, call_col);
    }
    int64_t m = A.tensor.shape[0];
    int64_t n = A.tensor.shape[1];
    if (m != 2 || n != 2) {
      throw util::Error("svd currently supports 2x2 matrices only", call_line, call_col);
    }
    double a = A.tensor.Data()[0], b = A.tensor.Data()[1], c = A.tensor.Data()[2],
           d = A.tensor.Data()[3];
    double s1s1 = a * a + c * c;
    double s2s2 = b * b + d * d;
    double off = a * b + c * d;
    double tr = s1s1 + s2s2;
    double det = s1s1 * s2s2 - off * off;
    double tmp = std::sqrt(std::max(0.0, tr * tr * 0.25 - det));
    double s1 = std::sqrt(std::max(0.0, tr * 0.5 + tmp));
    double s2 = std::sqrt(std::max(0.0, tr * 0.5 - tmp));
    std::vector<double> S = {s1, 0.0, 0.0, s2};
    // Compute V from eigenvectors of A^T A
    double ata00 = s1s1, ata01 = off, ata11 = s2s2;
    std::vector<double> v_flat(4, 0.0);
    if (std::abs(ata01) < 1e-12) {
      v_flat = {1, 0, 0, 1};
    } else {
      double t = (ata00 - ata11) / (2 * ata01);
      double sign = t >= 0 ? 1.0 : -1.0;
      double tau = sign / (std::abs(t) + std::sqrt(1 + t * t));
      double cs = 1 / std::sqrt(1 + tau * tau);
      double sn = cs * tau;
      v_flat = {cs, -sn, sn, cs};
    }
    std::vector<double> u_flat(4, 0.0);
    auto mul_AV = [&](const std::vector<double>& vec) -> std::vector<double> {
      return {a * vec[0] + b * vec[1], c * vec[0] + d * vec[1]};
    };
    auto norm2 = [](const std::vector<double>& v2) {
      return std::sqrt(v2[0] * v2[0] + v2[1] * v2[1]);
    };
    std::vector<double> v1 = {v_flat[0], v_flat[2]};
    std::vector<double> v2 = {v_flat[1], v_flat[3]};
    std::vector<double> u1 = mul_AV(v1);
    std::vector<double> u2 = mul_AV(v2);
    double n1 = norm2(u1);
    double n2 = norm2(u2);
    if (n1 > 0) {
      u_flat[0] = u1[0] / n1;
      u_flat[2] = u1[1] / n1;
    }
    if (n2 > 0) {
      u_flat[1] = u2[0] / n2;
      u_flat[3] = u2[1] / n2;
    }
    Value u_t = make_dense_tensor(u_flat, {2, 2}, A.tensor.elem_type);
    Value s_t = make_dense_tensor(S, {2, 2}, A.tensor.elem_type);
    Value v_t = make_dense_tensor(v_flat, {2, 2}, A.tensor.elem_type);
    return Value::Tuple({u_t, s_t, v_t});
  }
  if (name == "gamma") {
    expect_args(1, name);
    double x = ScalarF64(args[0], call_line, call_col);
    return Value::F64(std::tgamma(x));
  }
  if (name == "beta") {
    expect_args(2, name);
    double a = ScalarF64(args[0], call_line, call_col);
    double b = ScalarF64(args[1], call_line, call_col);
    return Value::F64(std::tgamma(a) * std::tgamma(b) / std::tgamma(a + b));
  }
  if (name == "erf") {
    expect_args(1, name);
    double x = ScalarF64(args[0], call_line, call_col);
    return Value::F64(std::erf(x));
  }
  if (name == "erfc") {
    expect_args(1, name);
    double x = ScalarF64(args[0], call_line, call_col);
    return Value::F64(std::erfc(x));
  }
  if (name == "igamma") {  // Regularized lower incomplete gamma P(a,x)
    expect_args(2, name);
    double a = ScalarF64(args[0], call_line, call_col);
    double x = ScalarF64(args[1], call_line, call_col);
    if (a <= 0 || x < 0) {
      throw util::Error("igamma expects a>0 and x>=0", call_line, call_col);
    }
    auto series = [&](double aa, double xx) {
      double sum = 1.0 / aa;
      double del = sum;
      for (int n = 1; n < 100; ++n) {
        del *= xx / (aa + n);
        sum += del;
        if (std::abs(del) < std::abs(sum) * 1e-12) break;
      }
      return sum * std::exp(-xx + aa * std::log(xx) - std::lgamma(aa));
    };
    auto contfrac = [&](double aa, double xx) {
      const double eps = 1e-12;
      const double fpmin = 1e-30;
      double b0 = 0.0;
      double b1 = 1.0;
      double fac = 1.0;
      double c1 = 1.0 / fpmin;
      double d1 = 1.0 / (xx - aa + 1.0);
      double h = d1;
      for (int i = 1; i < 1000; ++i) {
        double an = -static_cast<double>(i) * (static_cast<double>(i) - aa);
        double d = an * d1 + (xx + 2.0 * static_cast<double>(i) - aa);
        if (std::abs(d) < fpmin) d = fpmin;
        c1 = (xx + 2.0 * static_cast<double>(i) - aa) + an / c1;
        if (std::abs(c1) < fpmin) c1 = fpmin;
        d1 = 1.0 / d;
        double del = d1 * c1;
        h *= del;
        if (std::abs(del - 1.0) < eps) break;
      }
      return std::exp(-xx + aa * std::log(xx) - std::lgamma(aa)) * h;
    };
    double result;
    if (x < a + 1.0) {
      result = series(a, x);
    } else {
      result = 1.0 - contfrac(a, x);
    }
    return Value::F64(result);
  }
  if (name == "philox") {
    expect_args(3, name);
    uint64_t seed = static_cast<uint64_t>(CastTo(DType::kU64, args[0], call_line, call_col).u64);
    uint64_t stream = static_cast<uint64_t>(CastTo(DType::kU64, args[1], call_line, call_col).u64);
    uint64_t ctr = static_cast<uint64_t>(CastTo(DType::kU64, args[2], call_line, call_col).u64);
    uint64_t key = seed ^ (stream + 0x9E3779B97F4A7C15ULL);
    double out = Uint64ToUnitDouble(Philox2x32(ctr, key));
    return Value::F64(out);
  }
  if (name == "threefry") {
    expect_args(3, name);
    uint64_t seed = static_cast<uint64_t>(CastTo(DType::kU64, args[0], call_line, call_col).u64);
    uint64_t stream = static_cast<uint64_t>(CastTo(DType::kU64, args[1], call_line, call_col).u64);
    uint64_t ctr = static_cast<uint64_t>(CastTo(DType::kU64, args[2], call_line, call_col).u64);
    uint64_t key = seed ^ (stream + 0x9E3779B97F4A7C15ULL);
    double out = Uint64ToUnitDouble(Threefry2x32(ctr, key));
    return Value::F64(out);
  }
  if (name == "normal_pdf") {
    expect_args(3, name);
    double x = ScalarF64(args[0], call_line, call_col);
    double mu = ScalarF64(args[1], call_line, call_col);
    double sigma = ScalarF64(args[2], call_line, call_col);
    if (sigma <= 0.0) throw util::Error("normal_pdf sigma must be > 0", call_line, call_col);
    double z = (x - mu) / sigma;
    double coef = 1.0 / (sigma * std::sqrt(2.0 * M_PI));
    return Value::F64(coef * std::exp(-0.5 * z * z));
  }
  if (name == "normal_cdf") {
    expect_args(3, name);
    double x = ScalarF64(args[0], call_line, call_col);
    double mu = ScalarF64(args[1], call_line, call_col);
    double sigma = ScalarF64(args[2], call_line, call_col);
    if (sigma <= 0.0) throw util::Error("normal_cdf sigma must be > 0", call_line, call_col);
    double z = (x - mu) / (sigma * std::sqrt(2.0));
    return Value::F64(0.5 * (1.0 + std::erf(z)));
  }
  if (name == "normal_sample") {
    expect_args(5, name);
    double mu = ScalarF64(args[0], call_line, call_col);
    double sigma = ScalarF64(args[1], call_line, call_col);
    if (sigma <= 0.0) throw util::Error("normal_sample sigma must be > 0", call_line, call_col);
    uint64_t seed = static_cast<uint64_t>(CastTo(DType::kU64, args[2], call_line, call_col).u64);
    uint64_t stream = static_cast<uint64_t>(CastTo(DType::kU64, args[3], call_line, call_col).u64);
    uint64_t ctr = static_cast<uint64_t>(CastTo(DType::kU64, args[4], call_line, call_col).u64);
    uint64_t key = seed ^ (stream + 0x9E3779B97F4A7C15ULL);
    double u1 = Uint64ToUnitDouble(Philox2x32(ctr, key));
    double u2 = Uint64ToUnitDouble(Philox2x32(ctr + 1, key));
    double r = std::sqrt(-2.0 * std::log(std::max(u1, 1e-12)));
    double theta = 2.0 * M_PI * u2;
    double z = r * std::cos(theta);
    return Value::F64(mu + sigma * z);
  }
  if (name == "uniform_pdf") {
    expect_args(3, name);
    double x = ScalarF64(args[0], call_line, call_col);
    double a = ScalarF64(args[1], call_line, call_col);
    double b = ScalarF64(args[2], call_line, call_col);
    if (!(a < b)) throw util::Error("uniform_pdf requires a < b", call_line, call_col);
    if (x < a || x > b) return Value::F64(0.0);
    return Value::F64(1.0 / (b - a));
  }
  if (name == "uniform_cdf") {
    expect_args(3, name);
    double x = ScalarF64(args[0], call_line, call_col);
    double a = ScalarF64(args[1], call_line, call_col);
    double b = ScalarF64(args[2], call_line, call_col);
    if (!(a < b)) throw util::Error("uniform_cdf requires a < b", call_line, call_col);
    if (x <= a) return Value::F64(0.0);
    if (x >= b) return Value::F64(1.0);
    return Value::F64((x - a) / (b - a));
  }
  if (name == "uniform_sample") {
    expect_args(4, name);
    double a = ScalarF64(args[0], call_line, call_col);
    double b = ScalarF64(args[1], call_line, call_col);
    if (!(a < b)) throw util::Error("uniform_sample requires a < b", call_line, call_col);
    uint64_t seed = static_cast<uint64_t>(CastTo(DType::kU64, args[2], call_line, call_col).u64);
    uint64_t ctr = static_cast<uint64_t>(CastTo(DType::kU64, args[3], call_line, call_col).u64);
    double u = Uint64ToUnitDouble(Philox2x32(ctr, seed ^ 0x9E3779B97F4A7C15ULL));
    return Value::F64(a + (b - a) * u);
  }
  if (name == "exponential_pdf") {
    expect_args(2, name);
    double x = ScalarF64(args[0], call_line, call_col);
    double lambda = ScalarF64(args[1], call_line, call_col);
    if (lambda <= 0) throw util::Error("exponential_pdf lambda must be > 0", call_line, call_col);
    if (x < 0) throw util::Error("exponential_pdf domain requires x>=0", call_line, call_col);
    return Value::F64(lambda * std::exp(-lambda * x));
  }
  if (name == "exponential_cdf") {
    expect_args(2, name);
    double x = ScalarF64(args[0], call_line, call_col);
    double lambda = ScalarF64(args[1], call_line, call_col);
    if (lambda <= 0) throw util::Error("exponential_cdf lambda must be > 0", call_line, call_col);
    if (x < 0) throw util::Error("exponential_cdf domain requires x>=0", call_line, call_col);
    return Value::F64(1.0 - std::exp(-lambda * x));
  }
  if (name == "exponential_sample") {
    expect_args(3, name);
    double lambda = ScalarF64(args[0], call_line, call_col);
    if (lambda <= 0)
      throw util::Error("exponential_sample lambda must be > 0", call_line, call_col);
    uint64_t seed = static_cast<uint64_t>(CastTo(DType::kU64, args[1], call_line, call_col).u64);
    uint64_t ctr = static_cast<uint64_t>(CastTo(DType::kU64, args[2], call_line, call_col).u64);
    double u = std::max(Uint64ToUnitDouble(Philox2x32(ctr, seed ^ 0x517cc1b727220a95ULL)), 1e-12);
    return Value::F64(-std::log(u) / lambda);
  }
  if (name == "poisson_pmf") {
    expect_args(2, name);
    double k = ScalarF64(args[0], call_line, call_col);
    double lambda = ScalarF64(args[1], call_line, call_col);
    if (lambda <= 0) throw util::Error("poisson_pmf lambda must be > 0", call_line, call_col);
    if (k < 0 || std::floor(k) != k) {
      throw util::Error("poisson_pmf k must be non-negative integer", call_line, call_col);
    }
    double pmf = std::exp(-lambda + k * std::log(lambda) - std::lgamma(k + 1.0));
    return Value::F64(pmf);
  }
  if (name == "poisson_sample") {
    expect_args(3, name);
    double lambda = ScalarF64(args[0], call_line, call_col);
    if (lambda <= 0) throw util::Error("poisson_sample lambda must be > 0", call_line, call_col);
    uint64_t seed = static_cast<uint64_t>(CastTo(DType::kU64, args[1], call_line, call_col).u64);
    uint64_t ctr = static_cast<uint64_t>(CastTo(DType::kU64, args[2], call_line, call_col).u64);
    double L = std::exp(-lambda);
    double p = 1.0;
    int k = 0;
    uint64_t key = seed ^ 0x94d049bb133111ebULL;
    uint64_t local_ctr = ctr;
    do {
      ++k;
      p *= Uint64ToUnitDouble(Philox2x32(local_ctr++, key));
    } while (p > L && k < 1000000);
    return Value::I64(k - 1);
  }
  if (name == "binomial_pmf") {
    expect_args(3, name);
    double k = ScalarF64(args[0], call_line, call_col);
    double n = ScalarF64(args[1], call_line, call_col);
    double p = ScalarF64(args[2], call_line, call_col);
    if (p < 0 || p > 1) throw util::Error("binomial_pmf p must be in [0,1]", call_line, call_col);
    if (n < 0 || std::floor(n) != n)
      throw util::Error("binomial_pmf n must be integer >=0", call_line, call_col);
    if (k < 0 || k > n || std::floor(k) != k) {
      throw util::Error("binomial_pmf k must be integer in [0,n]", call_line, call_col);
    }
    double log_c = std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
    double pmf = std::exp(log_c + k * std::log(p) + (n - k) * std::log(1 - p));
    return Value::F64(pmf);
  }
  if (name == "binomial_sample") {
    expect_args(4, name);
    double n = ScalarF64(args[0], call_line, call_col);
    double p = ScalarF64(args[1], call_line, call_col);
    if (n < 0 || std::floor(n) != n)
      throw util::Error("binomial_sample n must be integer >=0", call_line, call_col);
    if (p < 0 || p > 1)
      throw util::Error("binomial_sample p must be in [0,1]", call_line, call_col);
    uint64_t seed = static_cast<uint64_t>(CastTo(DType::kU64, args[2], call_line, call_col).u64);
    uint64_t ctr = static_cast<uint64_t>(CastTo(DType::kU64, args[3], call_line, call_col).u64);
    uint64_t key = seed ^ 0xd9532fca9d4b5c15ULL;
    int64_t successes = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
      double u = Uint64ToUnitDouble(Philox2x32(ctr++, key));
      if (u < p) ++successes;
    }
    return Value::I64(successes);
  }
  if (name == "quantile") {
    expect_args(2, name);
    auto data = Tensor1DToVector(args[0], call_line, call_col);
    double q = ScalarF64(args[1], call_line, call_col);
    if (data.empty()) throw util::Error("quantile of empty data", call_line, call_col);
    if (q < 0.0 || q > 1.0) throw util::Error("quantile q must be in [0,1]", call_line, call_col);
    std::sort(data.begin(), data.end());
    double pos = q * (static_cast<double>(data.size() - 1));
    size_t idx = static_cast<size_t>(pos);
    double frac = pos - static_cast<double>(idx);
    double val = data[idx];
    if (idx + 1 < data.size()) val = val + frac * (data[idx + 1] - data[idx]);
    return Value::F64(val);
  }
  if (name == "correlation") {
    expect_args(2, name);
    auto x = Tensor1DToVector(args[0], call_line, call_col);
    auto y = Tensor1DToVector(args[1], call_line, call_col);
    if (x.size() != y.size() || x.empty()) {
      throw util::Error("correlation requires equal non-empty vectors", call_line, call_col);
    }
    double sumx = 0, sumy = 0, sumxx = 0, sumyy = 0, sumxy = 0;
    for (size_t i = 0; i < x.size(); ++i) {
      sumx += x[i];
      sumy += y[i];
      sumxx += x[i] * x[i];
      sumyy += y[i] * y[i];
      sumxy += x[i] * y[i];
    }
    double n = static_cast<double>(x.size());
    double num = n * sumxy - sumx * sumy;
    double den = std::sqrt((n * sumxx - sumx * sumx) * (n * sumyy - sumy * sumy));
    if (den == 0.0) throw util::Error("correlation undefined (zero variance)", call_line, call_col);
    return Value::F64(num / den);
  }
  if (name == "regression") {
    expect_args(2, name);
    auto x = Tensor1DToVector(args[0], call_line, call_col);
    auto y = Tensor1DToVector(args[1], call_line, call_col);
    if (x.size() != y.size() || x.empty()) {
      throw util::Error("regression requires equal non-empty vectors", call_line, call_col);
    }
    double sumx = 0, sumy = 0, sumxx = 0, sumxy = 0;
    for (size_t i = 0; i < x.size(); ++i) {
      sumx += x[i];
      sumy += y[i];
      sumxx += x[i] * x[i];
      sumxy += x[i] * y[i];
    }
    double n = static_cast<double>(x.size());
    double denom = (n * sumxx - sumx * sumx);
    if (denom == 0.0)
      throw util::Error("regression undefined (zero variance)", call_line, call_col);
    double slope = (n * sumxy - sumx * sumy) / denom;
    double intercept = (sumy - slope * sumx) / n;
    return Value::Tuple({Value::F64(slope), Value::F64(intercept)});
  }
  if (name == "keys") {
    expect_args(1, name);
    const Value& v = args[0];
    if (v.type != DType::kRecord) {
      throw util::Error("keys expects a record", call_line, call_col);
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
      throw util::Error("values expects a record", call_line, call_col);
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
      throw util::Error("has_key expects (record, string)", call_line, call_col);
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
    DType target = PromoteType(PromoteType(args[0].type, args[1].type, call_line, call_col),
                               args[2].type, call_line, call_col);
    Value v0 = CastTo(target, args[0], call_line, call_col);
    Value v1 = CastTo(target, args[1], call_line, call_col);
    Value v2 = CastTo(target, args[2], call_line, call_col);
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
    DType target = PromoteType(args[0].type, args[1].type, call_line, call_col);
    Value v0 = CastTo(target, args[0], call_line, call_col);
    Value v1 = CastTo(target, args[1], call_line, call_col);
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
    DType target = PromoteType(args[0].type, args[1].type, call_line, call_col);
    Value v0 = CastTo(target, args[0], call_line, call_col);
    Value v1 = CastTo(target, args[1], call_line, call_col);
    if (IsFloat(target)) {
      double res = std::max(v0.f64, v1.f64);
      return target == DType::kF32 ? Value::F32(static_cast<float>(res)) : Value::F64(res);
    }
    int64_t res = std::max(v0.i64, v1.i64);
    return Value::I64(res);
  }
  throw util::Error("Unknown function: " + name, call_line, call_col);
}

ExecResult Evaluator::EvaluateBlock(const parser::BlockStatement& block) {
  if (!env_) {
    throw util::Error("Environment is not configured", 0, 0);
  }
  auto block_env = std::make_shared<Environment>(env_);
  Evaluator block_eval(block_env);
  ExecResult last;
  for (const auto& stmt : block.statements) {
    ExecResult result = block_eval.EvaluateStatement(*stmt);
    if (result.control != ControlSignal::kNone) {
      return result;
    }
    last = result;
  }
  return last;
}

ExecResult Evaluator::EvaluateIf(const parser::IfStatement& stmt) {
  Value condition = Evaluate(*stmt.condition);
  if (RequireBool(condition, stmt.condition->line, stmt.condition->column)) {
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
    if (!RequireBool(condition, stmt.condition->line, stmt.condition->column)) {
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
  if (!env_) {
    throw util::Error("Environment is not configured", 0, 0);
  }
  auto loop_env = std::make_shared<Environment>(env_);
  Evaluator loop_eval(loop_env);
  ExecResult last;
  if (stmt.init) {
    ExecResult init_result = loop_eval.EvaluateStatement(*stmt.init);
    if (init_result.control != ControlSignal::kNone) {
      return init_result;
    }
  }
  while (true) {
    if (stmt.condition) {
      Value cond_val = loop_eval.Evaluate(*stmt.condition);
      if (!RequireBool(cond_val, stmt.condition->line, stmt.condition->column)) {
        break;
      }
    }
    ExecResult body = loop_eval.EvaluateStatement(*stmt.body);
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
      ExecResult inc = loop_eval.EvaluateStatement(*stmt.increment);
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
  if (!env_) {
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
  if (!env_) {
    throw util::Error("Environment is not configured", 0, 0);
  }
  auto bind_value = [&](const std::string& name, const Value& value) {
    if (!env_->Assign(name, value)) {
      env_->Define(name, value);
    }
  };
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
      bind_value(stmt.tuple_pattern->names[i], rhs.tuple.elements[i]);
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
      bind_value(stmt.record_pattern->fields[i].second, rhs.record.fields[idx].second);
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
  bind_value(stmt.name, value);
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
      std::string kind = "dense";
      switch (tensor.kind) {
        case TensorKind::kDense:
          kind = "dense";
          break;
        case TensorKind::kSparseCSR:
          kind = "sparse_csr";
          break;
        case TensorKind::kSparseCOO:
          kind = "sparse_coo";
          break;
        case TensorKind::kRagged:
          kind = "ragged";
          break;
      }
      oss << "tensor[" << kind << "]" << ShapeToString(tensor.shape) << "<"
          << DTypeToString(tensor.elem_type) << ">";
      auto render_dense = [&](const Value& dense) -> std::string {
        std::ostringstream out;
        std::function<void(size_t, int64_t)> rec = [&](size_t dim, int64_t offset) {
          if (dim + 1 == dense.tensor.shape.size()) {
            out << "[";
            for (int64_t i = 0; i < dense.tensor.shape[dim]; ++i) {
              if (i > 0) out << ", ";
              out << dense.tensor.Data()[offset + i * dense.tensor.strides[dim]];
            }
            out << "]";
            return;
          }
          out << "[";
          for (int64_t i = 0; i < dense.tensor.shape[dim]; ++i) {
            if (i > 0) out << ", ";
            rec(dim + 1, offset + i * dense.tensor.strides[dim]);
          }
          out << "]";
        };
        if (!dense.tensor.shape.empty()) rec(0, 0);
        return out.str();
      };
      constexpr int64_t kMaxElemsToPrint = 64;
      if (tensor.kind == TensorKind::kDense && tensor.size <= kMaxElemsToPrint) {
        oss << " " << render_dense(*this);
        return oss.str();
      }
      if (tensor.kind == TensorKind::kSparseCSR || tensor.kind == TensorKind::kSparseCOO) {
        oss << " nnz=" << tensor.sparse_values.size();
        if (tensor.sparse_values.size() <= static_cast<size_t>(kMaxElemsToPrint)) {
          oss << " values=[";
          if (tensor.kind == TensorKind::kSparseCSR) {
            for (size_t i = 0; i < tensor.sparse_values.size(); ++i) {
              if (i > 0) oss << ", ";
              oss << "(" << tensor.indices[i] << ":" << tensor.sparse_values[i] << ")";
            }
          } else {
            for (size_t i = 0; i < tensor.sparse_values.size(); ++i) {
              if (i > 0) oss << ", ";
              oss << "(" << tensor.rows[i] << "," << tensor.cols[i] << ":"
                  << tensor.sparse_values[i] << ")";
            }
          }
          oss << "]";
        }
        return oss.str();
      }
      if (tensor.kind == TensorKind::kRagged) {
        oss << " rows=" << (tensor.row_splits.empty() ? 0 : tensor.row_splits.size() - 1);
        if (tensor.ragged_values.size() <= static_cast<size_t>(kMaxElemsToPrint)) {
          oss << " values=[";
          for (size_t i = 0; i < tensor.ragged_values.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << tensor.ragged_values[i];
          }
          oss << "] splits=[";
          for (size_t i = 0; i < tensor.row_splits.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << tensor.row_splits[i];
          }
          oss << "]";
        }
        return oss.str();
      }
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
