#include "runtime/tensor_utils.h"

#include <sstream>
#include <stdexcept>

#include "util/error.h"

namespace lattice::runtime {

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

}  // namespace lattice::runtime
