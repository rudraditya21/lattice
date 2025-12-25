#ifndef LATTICE_RUNTIME_VALUE_H_
#define LATTICE_RUNTIME_VALUE_H_

#include <complex>
#include <cstdint>
#include <memory>
#include <numeric>
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
  std::vector<std::string> parameter_types;
  std::string return_type;
  std::unique_ptr<parser::Statement> body;
  Environment* defining_env;
};

struct Value {
  DType type;
  // Storage for different kinds; only some are active depending on type.
  int64_t i64 = 0;
  uint64_t u64 = 0;
  double f64 = 0.0;
  double number = 0.0;  // compatibility alias
  long double decimal = 0.0;
  struct Rational {
    int64_t num = 0;
    int64_t den = 1;
  } rational;
  std::complex<double> complex = {0.0, 0.0};
  bool boolean = false;
  std::shared_ptr<Function> function;
  struct TensorInfo {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    DType elem_type = DType::kF64;
    std::vector<double> storage;  // row-major, flattened
  } tensor;
  std::string type_name;

  /// Constructors for scalar types.
  static Value Bool(bool v) {
    Value val;
    val.type = DType::kBool;
    val.boolean = v;
    val.number = v ? 1.0 : 0.0;
    val.type_name = "bool";
    return val;
  }
  static Value I8(int8_t v);
  static Value I16(int16_t v);
  static Value I32(int32_t v);
  static Value I64(int64_t v);
  static Value U8(uint8_t v);
  static Value U16(uint16_t v);
  static Value U32(uint32_t v);
  static Value U64(uint64_t v);
  static Value F16(float v);
  static Value BF16(float v);
  static Value F32(float v);
  static Value F64(double v);
  static Value Complex64(std::complex<float> v);
  static Value Complex128(std::complex<double> v);
  static Value Decimal(long double v);
  static Value RationalValue(int64_t num, int64_t den);
  static Value RationalValueNormalized(int64_t num, int64_t den);
  static Value Tensor(std::vector<int64_t> shape, DType elem_type, double fill_value);
  /// Convenience constructor for function values.
  static Value Func(std::shared_ptr<Function> fn);
  /// Legacy convenience constructor for numeric values (defaults to f64).
  static Value Number(double v) { return F64(v); }

  /// Formats the value for display in the REPL.
  std::string ToString() const;
};

inline Value Value::I32(int32_t v) {
  Value val;
  val.type = DType::kI32;
  val.i64 = v;
  val.f64 = static_cast<double>(v);
  val.number = val.f64;
  val.type_name = "i32";
  return val;
}

inline Value Value::I64(int64_t v) {
  Value val;
  val.type = DType::kI64;
  val.i64 = v;
  val.f64 = static_cast<double>(v);
  val.number = val.f64;
  val.type_name = "i64";
  return val;
}

inline Value Value::U32(uint32_t v) {
  Value val;
  val.type = DType::kU32;
  val.u64 = v;
  val.f64 = static_cast<double>(v);
  val.number = val.f64;
  val.type_name = "u32";
  return val;
}

inline Value Value::U64(uint64_t v) {
  Value val;
  val.type = DType::kU64;
  val.u64 = v;
  val.f64 = static_cast<double>(v);
  val.number = val.f64;
  val.type_name = "u64";
  return val;
}

inline Value Value::I8(int8_t v) {
  Value val;
  val.type = DType::kI8;
  val.i64 = v;
  val.f64 = static_cast<double>(v);
  val.number = val.f64;
  val.type_name = "i8";
  return val;
}

inline Value Value::I16(int16_t v) {
  Value val;
  val.type = DType::kI16;
  val.i64 = v;
  val.f64 = static_cast<double>(v);
  val.number = val.f64;
  val.type_name = "i16";
  return val;
}

inline Value Value::U8(uint8_t v) {
  Value val;
  val.type = DType::kU8;
  val.u64 = v;
  val.f64 = static_cast<double>(v);
  val.number = val.f64;
  val.type_name = "u8";
  return val;
}

inline Value Value::U16(uint16_t v) {
  Value val;
  val.type = DType::kU16;
  val.u64 = v;
  val.f64 = static_cast<double>(v);
  val.number = val.f64;
  val.type_name = "u16";
  return val;
}

inline Value Value::F16(float v) {
  Value val;
  val.type = DType::kF16;
  val.f64 = static_cast<double>(v);
  val.number = val.f64;
  val.type_name = "f16";
  return val;
}

inline Value Value::BF16(float v) {
  Value val;
  val.type = DType::kBF16;
  val.f64 = static_cast<double>(v);
  val.number = val.f64;
  val.type_name = "bfloat16";
  return val;
}

inline Value Value::F32(float v) {
  Value val;
  val.type = DType::kF32;
  val.f64 = static_cast<double>(v);
  val.number = val.f64;
  val.type_name = "f32";
  return val;
}

inline Value Value::F64(double v) {
  Value val;
  val.type = DType::kF64;
  val.f64 = v;
  val.number = val.f64;
  val.type_name = "f64";
  return val;
}

inline Value Value::Complex64(std::complex<float> v) {
  Value val;
  val.type = DType::kC64;
  val.complex = std::complex<double>(v.real(), v.imag());
  val.type_name = "complex64";
  return val;
}

inline Value Value::Complex128(std::complex<double> v) {
  Value val;
  val.type = DType::kC128;
  val.complex = v;
  val.type_name = "complex128";
  return val;
}

inline Value Value::Decimal(long double v) {
  Value val;
  val.type = DType::kDecimal;
  val.decimal = v;
  val.number = static_cast<double>(v);
  val.type_name = "decimal";
  return val;
}

inline Value Value::RationalValue(int64_t num, int64_t den) {
  Value val;
  val.type = DType::kRational;
  val.rational.num = num;
  val.rational.den = den == 0 ? 1 : den;
  val.f64 = static_cast<double>(num) / static_cast<double>(den == 0 ? 1 : den);
  val.number = val.f64;
  val.type_name = "rational";
  return val;
}

inline Value Value::RationalValueNormalized(int64_t num, int64_t den) {
  if (den == 0) den = 1;
  int64_t g = std::gcd(num, den);
  num /= g;
  den /= g;
  if (den < 0) {
    num = -num;
    den = -den;
  }
  return RationalValue(num, den);
}

inline Value Value::Func(std::shared_ptr<Function> fn) {
  Value val;
  val.type = DType::kFunction;
  val.function = std::move(fn);
  val.number = 0.0;
  val.type_name = "function";
  return val;
}

inline Value Value::Tensor(std::vector<int64_t> shape, DType elem_type, double fill_value) {
  Value val;
  val.type = DType::kTensor;
  val.tensor.shape = std::move(shape);
  val.tensor.elem_type = elem_type;
  val.type_name = "tensor";
  // Compute strides (row-major).
  val.tensor.strides.resize(val.tensor.shape.size());
  int64_t stride = 1;
  for (int i = static_cast<int>(val.tensor.shape.size()) - 1; i >= 0; --i) {
    val.tensor.strides[i] = stride;
    stride *= val.tensor.shape[i];
  }
  const int64_t total = stride;
  val.tensor.storage.assign(static_cast<size_t>(total), fill_value);
  val.number = 0.0;
  return val;
}

}  // namespace runtime
}  // namespace lattice

#endif  // LATTICE_RUNTIME_VALUE_H_
