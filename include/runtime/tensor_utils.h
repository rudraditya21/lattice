#ifndef LATTICE_RUNTIME_TENSOR_UTILS_H_
#define LATTICE_RUNTIME_TENSOR_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "runtime/value.h"

namespace lattice::runtime {

std::string ShapeToString(const std::vector<int64_t>& shape);

std::optional<std::vector<int64_t>> BroadcastShape(const std::vector<int64_t>& a,
                                                   const std::vector<int64_t>& b);
std::vector<int64_t> BroadcastStrides(const std::vector<int64_t>& shape,
                                      const std::vector<int64_t>& strides, size_t out_rank);
int64_t OffsetFromFlatIndex(int64_t flat, const std::vector<int64_t>& out_strides,
                            const std::vector<int64_t>& broadcast_strides);

Value ToDenseTensor(const Value& v, int line, int column);
Value DenseToCSR(const Value& dense);
Value DenseToCOO(const Value& dense);

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_TENSOR_UTILS_H_
