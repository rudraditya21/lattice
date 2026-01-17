#include "test_util.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "parser/ast.h"
#include "runtime/tensor_gpu.h"

namespace test {

namespace {

bool EnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return false;
  }
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

rt::Value MakeTensor(const std::vector<int64_t>& shape, const std::vector<double>& data) {
  rt::Value v = rt::Value::Tensor(shape, rt::DType::kF64, 0.0);
  const size_t count = static_cast<size_t>(v.tensor.size);
  if (count != data.size()) {
    return v;
  }
  for (size_t i = 0; i < count; ++i) {
    v.tensor.Data()[static_cast<int64_t>(i)] = data[i];
  }
  v.tensor.elem_type = rt::DType::kF64;
  return v;
}

bool TensorNear(const rt::Value& v,
                const std::vector<double>& expected,
                double tol) {
  if (v.type != rt::DType::kTensor) {
    return false;
  }
  if (static_cast<size_t>(v.tensor.size) != expected.size()) {
    return false;
  }
  const double* data = v.tensor.Data();
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(data[i] - expected[i]) > tol) {
      return false;
    }
  }
  return true;
}

}  // namespace

void RunGpuCorrectnessTests(TestContext* ctx) {
  if (!EnvEnabled("LATTICE_GPU_CORRECTNESS_TEST")) {
    return;
  }
  const auto* backend = rt::GetDefaultBackend();
  if (!backend || backend->Type() == rt::BackendType::kCPU) {
    return;
  }

  const double tol = 1e-3;
  std::string error;

  // Elementwise add.
  auto lhs = MakeTensor({2, 2}, {1, 2, 3, 4});
  auto rhs = MakeTensor({2, 2}, {5, 6, 7, 8});
  auto add_or =
      rt::TryGpuElemwise(lhs, rhs, ps::BinaryOp::kAdd, 1, 1, &error);
  ExpectTrue(add_or.has_value(), "gpu_elemwise_add_available", ctx);
  if (add_or) {
    ExpectTrue(TensorNear(add_or.value(), {6, 8, 10, 12}, tol),
               "gpu_elemwise_add_correct", ctx);
  }

  // Reduce sum.
  auto sum_or = rt::TryGpuReduce(lhs, rt::ReduceKind::kSum, 1, 1, &error);
  ExpectTrue(sum_or.has_value(), "gpu_reduce_sum_available", ctx);
  if (sum_or) {
    ExpectTrue(std::fabs(sum_or->f64 - 10.0) <= tol, "gpu_reduce_sum_correct", ctx);
  }

  // Transpose.
  auto t_in = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
  auto tr_or = rt::TryGpuTranspose(t_in, 1, 1, &error);
  ExpectTrue(tr_or.has_value(), "gpu_transpose_available", ctx);
  if (tr_or) {
    ExpectTrue(TensorNear(tr_or.value(), {1, 4, 2, 5, 3, 6}, tol),
               "gpu_transpose_correct", ctx);
  }

  // Matmul.
  auto a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
  auto b = MakeTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  auto mm_or = rt::TryGpuMatmul(a, b, 1, 1, &error);
  ExpectTrue(mm_or.has_value(), "gpu_matmul_available", ctx);
  if (mm_or) {
    ExpectTrue(TensorNear(mm_or.value(), {22, 28, 49, 64}, tol),
               "gpu_matmul_correct", ctx);
  }

  // Conv2d.
  auto input = MakeTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto kernel = MakeTensor({2, 2}, {1, 0, 0, 1});
  auto conv_or = rt::TryGpuConv2d(input, kernel, 1, 1, &error);
  ExpectTrue(conv_or.has_value(), "gpu_conv2d_available", ctx);
  if (conv_or) {
    ExpectTrue(TensorNear(conv_or.value(), {6, 8, 12, 14}, tol),
               "gpu_conv2d_correct", ctx);
  }
}

}  // namespace test
