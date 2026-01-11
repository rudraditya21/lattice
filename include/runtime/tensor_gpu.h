#ifndef LATTICE_RUNTIME_TENSOR_GPU_H_
#define LATTICE_RUNTIME_TENSOR_GPU_H_

#include <cstdint>
#include <optional>
#include <string>

#include "parser/ast.h"
#include "runtime/value.h"

namespace lattice::runtime {

enum class ReduceKind { kSum, kMean, kVar, kStd };

std::optional<Value> TryGpuElemwise(const Value& lhs, const Value& rhs, parser::BinaryOp op,
                                    int line, int column, std::string* error);
std::optional<Value> TryGpuReduce(const Value& v, ReduceKind kind, int line, int column,
                                  std::string* error);
std::optional<Value> TryGpuTranspose(const Value& v, int line, int column, std::string* error);
std::optional<Value> TryGpuMatmul(const Value& lhs, const Value& rhs, int line, int column,
                                  std::string* error);
std::optional<Value> TryGpuConv2d(const Value& input, const Value& kernel, int line, int column,
                                  std::string* error);
std::optional<Value> TryGpuMaxPool2d(const Value& input, int64_t k_h, int64_t k_w, int line,
                                     int column, std::string* error);
std::optional<Value> TryGpuFft1d(const Value& input, int line, int column, std::string* error);
std::optional<Value> TryGpuSolve(const Value& a, const Value& b, int line, int column,
                                 std::string* error);
std::optional<Value> TryGpuLu(const Value& a, int line, int column, std::string* error);
std::optional<Value> TryGpuQr(const Value& a, int line, int column, std::string* error);
std::optional<Value> TryGpuSvd(const Value& a, int line, int column, std::string* error);
std::optional<Value> TryGpuQuantile(const Value& data, double q, int line, int column,
                                    std::string* error);
std::optional<Value> TryGpuCorrelation(const Value& x, const Value& y, int line, int column,
                                       std::string* error);
std::optional<Value> TryGpuRegression(const Value& x, const Value& y, int line, int column,
                                      std::string* error);

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_TENSOR_GPU_H_
