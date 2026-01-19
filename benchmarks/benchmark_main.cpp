#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(LATTICE_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(LATTICE_USE_CBLAS)
#include <cblas.h>
#endif

#include "runtime/backend.h"
#include "runtime/backends/cuda_backend.h"
#include "runtime/backends/hip_backend.h"
#include "runtime/backends/opencl_backend.h"
#include "runtime/tensor_gpu.h"
#if defined(__APPLE__)
#include "runtime/backends/metal_backend.h"
#endif

namespace {

struct BenchOptions {
  int warmup = 3;
  int iters = 10;
  bool check_baseline = false;
  bool tune = false;
  std::string baseline_path = "benchmarks/baselines.txt";
  std::string output_path;
  std::vector<std::string> ops;
  int64_t matmul_m = 512;
  int64_t matmul_k = 512;
  int64_t matmul_n = 512;
};

struct BenchResult {
  std::string backend;
  std::string device_class;
  std::string device_name;
  std::string op;
  std::string shape;
  std::string metric;
  std::string unit;
  double value = 0.0;
};

struct BaselineEntry {
  std::string device_class;
  std::string op;
  std::string shape;
  std::string metric;
  double baseline = 0.0;
  double tolerance = 0.1;
};

bool IsTrue(const std::string& value) {
  std::string v = value;
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

bool SetEnvVar(const std::string& name, const std::string& value) {
#if defined(_WIN32)
  return _putenv_s(name.c_str(), value.c_str()) == 0;
#else
  return setenv(name.c_str(), value.c_str(), 1) == 0;
#endif
}

std::string Lower(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  for (char c : input) {
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

std::vector<std::string> SplitOps(const std::string& input) {
  std::vector<std::string> out;
  std::string token;
  for (char c : input) {
    if (c == ',') {
      if (!token.empty()) {
        out.push_back(Lower(token));
        token.clear();
      }
      continue;
    }
    if (!std::isspace(static_cast<unsigned char>(c))) {
      token.push_back(c);
    }
  }
  if (!token.empty()) {
    out.push_back(Lower(token));
  }
  return out;
}

bool ShouldRunOp(const BenchOptions& opts, const std::string& op) {
  if (opts.ops.empty()) {
    return true;
  }
  const std::string key = Lower(op);
  for (const auto& entry : opts.ops) {
    if (entry == key) {
      return true;
    }
  }
  return false;
}

bool ParseMatmulShape(const std::string& input,
                      int64_t* m,
                      int64_t* k,
                      int64_t* n) {
  std::vector<int64_t> parts;
  std::string token;
  for (char c : input) {
    if (c == 'x' || c == 'X') {
      if (token.empty()) {
        return false;
      }
      parts.push_back(std::strtoll(token.c_str(), nullptr, 10));
      token.clear();
      continue;
    }
    token.push_back(c);
  }
  if (!token.empty()) {
    parts.push_back(std::strtoll(token.c_str(), nullptr, 10));
  }
  if (parts.size() == 1) {
    *m = parts[0];
    *k = parts[0];
    *n = parts[0];
    return true;
  }
  if (parts.size() == 3) {
    *m = parts[0];
    *k = parts[1];
    *n = parts[2];
    return true;
  }
  return false;
}

std::string SanitizeToken(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  for (char c : input) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-') {
      out.push_back(static_cast<char>(std::tolower(c)));
    } else if (c == ' ') {
      out.push_back('_');
    }
  }
  if (out.empty()) {
    return "unknown";
  }
  return out;
}

std::string DeviceClassForBackend(const lattice::runtime::Backend* backend) {
  using lattice::runtime::BackendType;
  if (!backend) {
    return "unknown";
  }
  switch (backend->Type()) {
    case BackendType::kCPU:
      return "cpu";
    case BackendType::kCUDA: {
      const auto* cuda =
          static_cast<const lattice::runtime::CudaBackend*>(backend);
      auto info = cuda->DeviceInfo();
      if (info.empty()) {
        return "cuda_unknown";
      }
      const auto& dev = info.front();
      std::ostringstream out;
      out << "cuda_sm" << dev.major << dev.minor;
      return out.str();
    }
    case BackendType::kHIP: {
      const auto* hip =
          static_cast<const lattice::runtime::HipBackend*>(backend);
      auto info = hip->DeviceInfo();
      if (info.empty()) {
        return "hip_unknown";
      }
      return "hip_" + SanitizeToken(info.front().name);
    }
    case BackendType::kOpenCL: {
      const auto* opencl =
          static_cast<const lattice::runtime::OpenCLBackend*>(backend);
      auto info = opencl->DeviceInfo();
      if (info.empty()) {
        return "opencl_unknown";
      }
      return "opencl_" + SanitizeToken(info.front().vendor);
    }
    case BackendType::kMetal: {
#if defined(__APPLE__)
      const auto* metal =
          static_cast<const lattice::runtime::MetalBackend*>(backend);
      auto info = metal->DeviceInfo();
      if (info.empty()) {
        return "metal_unknown";
      }
      return "metal_" + SanitizeToken(info.front().name);
#else
      return "metal_unavailable";
#endif
    }
  }
  return "unknown";
}

std::string BackendNameForType(lattice::runtime::BackendType type) {
  using lattice::runtime::BackendType;
  switch (type) {
    case BackendType::kCPU:
      return "cpu";
    case BackendType::kCUDA:
      return "cuda";
    case BackendType::kHIP:
      return "hip";
    case BackendType::kOpenCL:
      return "opencl";
    case BackendType::kMetal:
      return "metal";
  }
  return "cpu";
}

bool CpuMatmul(const std::vector<double>& lhs,
               const std::vector<double>& rhs,
               std::vector<double>* out,
               int64_t m,
               int64_t n,
               int64_t k) {
  if (!out) {
    return false;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return true;
  }
#if defined(LATTICE_USE_ACCELERATE) || defined(LATTICE_USE_CBLAS)
  if (m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max() ||
      k > std::numeric_limits<int>::max()) {
    return false;
  }
  const double alpha = 1.0;
  const double beta = 0.0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              static_cast<int>(m), static_cast<int>(n),
              static_cast<int>(k), alpha, lhs.data(), static_cast<int>(k),
              rhs.data(), static_cast<int>(n), beta, out->data(),
              static_cast<int>(n));
  return true;
#else
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int64_t kk = 0; kk < k; ++kk) {
        acc += lhs[static_cast<size_t>(i * k + kk)] *
               rhs[static_cast<size_t>(kk * n + j)];
      }
      (*out)[static_cast<size_t>(i * n + j)] = acc;
    }
  }
  return true;
#endif
}

std::string DeviceNameForBackend(const lattice::runtime::Backend* backend) {
  using lattice::runtime::BackendType;
  if (!backend) {
    return "";
  }
  switch (backend->Type()) {
    case BackendType::kCPU:
      return "cpu";
    case BackendType::kCUDA: {
      const auto* cuda =
          static_cast<const lattice::runtime::CudaBackend*>(backend);
      auto info = cuda->DeviceInfo();
      return info.empty() ? "" : info.front().name;
    }
    case BackendType::kHIP: {
      const auto* hip =
          static_cast<const lattice::runtime::HipBackend*>(backend);
      auto info = hip->DeviceInfo();
      return info.empty() ? "" : info.front().name;
    }
    case BackendType::kOpenCL: {
      const auto* opencl =
          static_cast<const lattice::runtime::OpenCLBackend*>(backend);
      auto info = opencl->DeviceInfo();
      return info.empty() ? "" : info.front().name;
    }
    case BackendType::kMetal: {
#if defined(__APPLE__)
      const auto* metal =
          static_cast<const lattice::runtime::MetalBackend*>(backend);
      auto info = metal->DeviceInfo();
      return info.empty() ? "" : info.front().name;
#else
      return "";
#endif
    }
  }
  return "";
}

double MeasureNs(int warmup, int iters, const std::function<bool()>& fn) {
  for (int i = 0; i < warmup; ++i) {
    if (!fn()) {
      return -1.0;
    }
  }
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    if (!fn()) {
      return -1.0;
    }
  }
  auto end = std::chrono::steady_clock::now();
  auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
          .count();
  return static_cast<double>(ns) / static_cast<double>(iters);
}

std::vector<BaselineEntry> LoadBaselines(const std::string& path) {
  std::vector<BaselineEntry> out;
  std::ifstream in(path);
  if (!in) {
    return out;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    std::string token;
    BaselineEntry entry;
    bool ok = true;
    while (iss >> token) {
      auto pos = token.find('=');
      if (pos == std::string::npos) {
        continue;
      }
      const std::string key = token.substr(0, pos);
      const std::string value = token.substr(pos + 1);
      if (key == "device_class") {
        entry.device_class = value;
      } else if (key == "op") {
        entry.op = value;
      } else if (key == "shape") {
        entry.shape = value;
      } else if (key == "metric") {
        entry.metric = value;
      } else if (key == "baseline") {
        entry.baseline = std::strtod(value.c_str(), nullptr);
      } else if (key == "tol") {
        entry.tolerance = std::strtod(value.c_str(), nullptr);
      }
    }
    if (entry.device_class.empty() || entry.op.empty() ||
        entry.shape.empty() || entry.metric.empty()) {
      ok = false;
    }
    if (ok) {
      out.push_back(entry);
    }
  }
  return out;
}

const BaselineEntry* FindBaseline(const std::vector<BaselineEntry>& entries,
                                  const BenchResult& result) {
  for (const auto& entry : entries) {
    if (entry.device_class == result.device_class && entry.op == result.op &&
        entry.shape == result.shape && entry.metric == result.metric) {
      return &entry;
    }
  }
  return nullptr;
}

bool RegressionDetected(const BenchResult& result, const BaselineEntry& entry) {
  if (entry.baseline <= 0.0) {
    return false;
  }
  const bool latency = result.metric.find("latency") != std::string::npos;
  if (latency) {
    return result.value > entry.baseline * (1.0 + entry.tolerance);
  }
  return result.value < entry.baseline * (1.0 - entry.tolerance);
}

std::string ResultLine(const BenchResult& result) {
  std::ostringstream out;
  out << "backend=" << result.backend;
  out << " device_class=" << result.device_class;
  if (!result.device_name.empty()) {
    out << " device=\"" << result.device_name << "\"";
  }
  out << " op=" << result.op;
  out << " shape=" << result.shape;
  out << " metric=" << result.metric;
  out << " value=" << result.value;
  out << " unit=" << result.unit;
  return out.str();
}

void AppendResult(std::vector<BenchResult>* results,
                  const std::string& op,
                  const std::string& shape,
                  const std::string& metric,
                  const std::string& unit,
                  double value,
                  const lattice::runtime::Backend* backend) {
  BenchResult out;
  out.backend = BackendNameForType(backend->Type());
  out.device_class = DeviceClassForBackend(backend);
  out.device_name = DeviceNameForBackend(backend);
  out.op = op;
  out.shape = shape;
  out.metric = metric;
  out.unit = unit;
  out.value = value;
  results->push_back(out);
}

BenchOptions ParseArgs(int argc, char** argv) {
  BenchOptions opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--backend" && i + 1 < argc) {
      SetEnvVar("LATTICE_BACKEND", argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      opts.warmup = std::atoi(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      opts.iters = std::atoi(argv[++i]);
    } else if (arg == "--baseline" && i + 1 < argc) {
      opts.baseline_path = argv[++i];
    } else if (arg == "--check-baseline") {
      opts.check_baseline = true;
    } else if (arg == "--tune") {
      opts.tune = true;
    } else if (arg == "--output" && i + 1 < argc) {
      opts.output_path = argv[++i];
    } else if (arg == "--ops" && i + 1 < argc) {
      opts.ops = SplitOps(argv[++i]);
    } else if (arg == "--matmul-size" && i + 1 < argc) {
      int64_t size = std::strtoll(argv[++i], nullptr, 10);
      if (size > 0) {
        opts.matmul_m = size;
        opts.matmul_k = size;
        opts.matmul_n = size;
      }
    } else if (arg == "--matmul-shape" && i + 1 < argc) {
      int64_t m = 0;
      int64_t k = 0;
      int64_t n = 0;
      if (ParseMatmulShape(argv[++i], &m, &k, &n)) {
        opts.matmul_m = m;
        opts.matmul_k = k;
        opts.matmul_n = n;
      }
    }
  }
  return opts;
}

lattice::runtime::Value MakeTensor(const std::vector<int64_t>& shape,
                                   double fill) {
  return lattice::runtime::Value::Tensor(shape, lattice::runtime::DType::kF64,
                                         fill);
}

void RunMicrobench(const BenchOptions& opts,
                   std::vector<BenchResult>* results,
                   const lattice::runtime::Backend* backend) {
  using lattice::runtime::ReduceKind;
  using lattice::runtime::TryGpuConv2d;
  using lattice::runtime::TryGpuElemwise;
  using lattice::runtime::TryGpuMatmul;
  using lattice::runtime::TryGpuReduce;
  using lattice::runtime::TryGpuTranspose;
  using lattice::parser::BinaryOp;

  std::string error;
  const size_t n = 1 << 20;
  const double elem_bytes = sizeof(double);

  if (ShouldRunOp(opts, "elemwise")) {
    auto elemwise_fn = [&]() {
      auto lhs = MakeTensor({static_cast<int64_t>(n)}, 1.0);
      auto rhs = MakeTensor({static_cast<int64_t>(n)}, 2.0);
      auto out = TryGpuElemwise(lhs, rhs, BinaryOp::kAdd, 1, 1, &error);
      return out.has_value();
    };
    const double elemwise_ns = MeasureNs(opts.warmup, opts.iters, elemwise_fn);
    if (elemwise_ns > 0) {
      const double bytes = 3.0 * elem_bytes * static_cast<double>(n);
      const double flops = static_cast<double>(n);
      const double sec = elemwise_ns * 1e-9;
      AppendResult(results, "elemwise", "n=1048576", "latency_us", "us",
                   elemwise_ns / 1000.0, backend);
      AppendResult(results, "elemwise", "n=1048576", "bandwidth_gbps", "GB/s",
                   (bytes / sec) / 1e9, backend);
      AppendResult(results, "elemwise", "n=1048576", "gflops", "GFLOP/s",
                   (flops / sec) / 1e9, backend);
    }
  }

  if (ShouldRunOp(opts, "reduce_sum")) {
    auto reduce_fn = [&]() {
      auto lhs = MakeTensor({static_cast<int64_t>(n)}, 1.0);
      auto out = TryGpuReduce(lhs, ReduceKind::kSum, 1, 1, &error);
      return out.has_value();
    };
    const double reduce_ns = MeasureNs(opts.warmup, opts.iters, reduce_fn);
    if (reduce_ns > 0) {
      const double bytes = elem_bytes * static_cast<double>(n);
      const double flops = static_cast<double>(n);
      const double sec = reduce_ns * 1e-9;
      AppendResult(results, "reduce_sum", "n=1048576", "latency_us", "us",
                   reduce_ns / 1000.0, backend);
      AppendResult(results, "reduce_sum", "n=1048576", "bandwidth_gbps", "GB/s",
                   (bytes / sec) / 1e9, backend);
      AppendResult(results, "reduce_sum", "n=1048576", "gflops", "GFLOP/s",
                   (flops / sec) / 1e9, backend);
    }
  }

  const int64_t rows = 1024;
  const int64_t cols = 1024;
  if (ShouldRunOp(opts, "transpose")) {
    auto transpose_fn = [&]() {
      auto lhs = MakeTensor({rows, cols}, 1.0);
      auto out = TryGpuTranspose(lhs, 1, 1, &error);
      return out.has_value();
    };
    const double transpose_ns =
        MeasureNs(opts.warmup, opts.iters, transpose_fn);
    if (transpose_ns > 0) {
      const double count = static_cast<double>(rows * cols);
      const double bytes = 2.0 * elem_bytes * count;
      const double sec = transpose_ns * 1e-9;
      AppendResult(results, "transpose", "1024x1024", "latency_us", "us",
                   transpose_ns / 1000.0, backend);
      AppendResult(results, "transpose", "1024x1024", "bandwidth_gbps", "GB/s",
                   (bytes / sec) / 1e9, backend);
    }
  }

  const int64_t m = opts.matmul_m;
  const int64_t k = opts.matmul_k;
  const int64_t nn = opts.matmul_n;
  if (ShouldRunOp(opts, "matmul")) {
    auto matmul_fn = [&]() {
      auto lhs = MakeTensor({m, k}, 1.0);
      auto rhs = MakeTensor({k, nn}, 2.0);
      auto out = TryGpuMatmul(lhs, rhs, 1, 1, &error);
      return out.has_value();
    };
    const double matmul_ns = MeasureNs(opts.warmup, opts.iters, matmul_fn);
    if (matmul_ns > 0) {
      const double bytes =
          elem_bytes * static_cast<double>(m * k + k * nn + m * nn);
      const double flops = 2.0 * static_cast<double>(m * k * nn);
      const double sec = matmul_ns * 1e-9;
      std::ostringstream shape;
      shape << m << "x" << k << "x" << nn;
      AppendResult(results, "matmul", shape.str(), "latency_us", "us",
                   matmul_ns / 1000.0, backend);
      AppendResult(results, "matmul", shape.str(), "bandwidth_gbps", "GB/s",
                   (bytes / sec) / 1e9, backend);
      AppendResult(results, "matmul", shape.str(), "gflops", "GFLOP/s",
                   (flops / sec) / 1e9, backend);
    }
  }

  const int64_t h = 128;
  const int64_t w = 128;
  const int64_t kh = 3;
  const int64_t kw = 3;
  if (ShouldRunOp(opts, "conv2d")) {
    auto conv_fn = [&]() {
      auto input = MakeTensor({h, w}, 1.0);
      auto kernel = MakeTensor({kh, kw}, 1.0);
      auto out = TryGpuConv2d(input, kernel, 1, 1, &error);
      return out.has_value();
    };
    const double conv_ns = MeasureNs(opts.warmup, opts.iters, conv_fn);
    if (conv_ns > 0) {
      const int64_t out_h = h - kh + 1;
      const int64_t out_w = w - kw + 1;
      const double bytes = elem_bytes * static_cast<double>(
                                            h * w + kh * kw + out_h * out_w);
      const double flops =
          2.0 * static_cast<double>(out_h * out_w * kh * kw);
      const double sec = conv_ns * 1e-9;
      AppendResult(results, "conv2d", "128x128_k3", "latency_us", "us",
                   conv_ns / 1000.0, backend);
      AppendResult(results, "conv2d", "128x128_k3", "bandwidth_gbps", "GB/s",
                   (bytes / sec) / 1e9, backend);
      AppendResult(results, "conv2d", "128x128_k3", "gflops", "GFLOP/s",
                   (flops / sec) / 1e9, backend);
    }
  }
}

void RunCpuMicrobench(const BenchOptions& opts,
                      std::vector<BenchResult>* results,
                      const lattice::runtime::Backend* backend) {
  const size_t n = 1 << 20;
  const double elem_bytes = sizeof(double);
  volatile double sink = 0.0;

  std::vector<double> a(n, 1.0);
  std::vector<double> b(n, 2.0);
  std::vector<double> out(n, 0.0);

  if (ShouldRunOp(opts, "elemwise")) {
    auto elemwise_fn = [&]() {
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
      }
      sink += out[0];
      return true;
    };
    const double elemwise_ns = MeasureNs(opts.warmup, opts.iters, elemwise_fn);
    if (elemwise_ns > 0) {
      const double bytes = 3.0 * elem_bytes * static_cast<double>(n);
      const double flops = static_cast<double>(n);
      const double sec = elemwise_ns * 1e-9;
      AppendResult(results, "elemwise", "n=1048576", "latency_us", "us",
                   elemwise_ns / 1000.0, backend);
      AppendResult(results, "elemwise", "n=1048576", "bandwidth_gbps", "GB/s",
                   (bytes / sec) / 1e9, backend);
      AppendResult(results, "elemwise", "n=1048576", "gflops", "GFLOP/s",
                   (flops / sec) / 1e9, backend);
    }
  }

  if (ShouldRunOp(opts, "reduce_sum")) {
    auto reduce_fn = [&]() {
      double acc = 0.0;
      for (size_t i = 0; i < n; ++i) {
        acc += a[i];
      }
      sink += acc;
      return true;
    };
    const double reduce_ns = MeasureNs(opts.warmup, opts.iters, reduce_fn);
    if (reduce_ns > 0) {
      const double bytes = elem_bytes * static_cast<double>(n);
      const double flops = static_cast<double>(n);
      const double sec = reduce_ns * 1e-9;
      AppendResult(results, "reduce_sum", "n=1048576", "latency_us", "us",
                   reduce_ns / 1000.0, backend);
      AppendResult(results, "reduce_sum", "n=1048576", "bandwidth_gbps", "GB/s",
                   (bytes / sec) / 1e9, backend);
      AppendResult(results, "reduce_sum", "n=1048576", "gflops", "GFLOP/s",
                   (flops / sec) / 1e9, backend);
    }
  }

  const int64_t rows = 1024;
  const int64_t cols = 1024;
  if (ShouldRunOp(opts, "transpose")) {
    std::vector<double> in(static_cast<size_t>(rows * cols), 1.0);
    std::vector<double> trans(static_cast<size_t>(rows * cols), 0.0);
    auto transpose_fn = [&]() {
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          trans[static_cast<size_t>(c * rows + r)] =
              in[static_cast<size_t>(r * cols + c)];
        }
      }
      sink += trans[0];
      return true;
    };
    const double transpose_ns =
        MeasureNs(opts.warmup, opts.iters, transpose_fn);
    if (transpose_ns > 0) {
      const double count = static_cast<double>(rows * cols);
      const double bytes = 2.0 * elem_bytes * count;
      const double sec = transpose_ns * 1e-9;
      AppendResult(results, "transpose", "1024x1024", "latency_us", "us",
                   transpose_ns / 1000.0, backend);
      AppendResult(results, "transpose", "1024x1024", "bandwidth_gbps", "GB/s",
                   (bytes / sec) / 1e9, backend);
    }
  }

  const int64_t m = opts.matmul_m;
  const int64_t k = opts.matmul_k;
  const int64_t nn = opts.matmul_n;
  if (ShouldRunOp(opts, "matmul")) {
    std::vector<double> lhs(static_cast<size_t>(m * k), 1.0);
    std::vector<double> rhs(static_cast<size_t>(k * nn), 2.0);
    std::vector<double> out_mat(static_cast<size_t>(m * nn), 0.0);
    auto matmul_fn = [&]() {
      if (!CpuMatmul(lhs, rhs, &out_mat, m, nn, k)) {
        return false;
      }
      sink += out_mat[0];
      return true;
    };
    const double matmul_ns = MeasureNs(opts.warmup, opts.iters, matmul_fn);
    if (matmul_ns > 0) {
      const double bytes =
          elem_bytes * static_cast<double>(m * k + k * nn + m * nn);
      const double flops = 2.0 * static_cast<double>(m * k * nn);
      const double sec = matmul_ns * 1e-9;
      std::ostringstream shape;
      shape << m << "x" << k << "x" << nn;
      AppendResult(results, "matmul", shape.str(), "latency_us", "us",
                   matmul_ns / 1000.0, backend);
      AppendResult(results, "matmul", shape.str(), "bandwidth_gbps", "GB/s",
                   (bytes / sec) / 1e9, backend);
      AppendResult(results, "matmul", shape.str(), "gflops", "GFLOP/s",
                   (flops / sec) / 1e9, backend);
    }
  }

  const int64_t h = 128;
  const int64_t w = 128;
  const int64_t kh = 3;
  const int64_t kw = 3;
  if (ShouldRunOp(opts, "conv2d")) {
    std::vector<double> input(static_cast<size_t>(h * w), 1.0);
    std::vector<double> kernel(static_cast<size_t>(kh * kw), 1.0);
    const int64_t out_h = h - kh + 1;
    const int64_t out_w = w - kw + 1;
    std::vector<double> out_conv(static_cast<size_t>(out_h * out_w), 0.0);
    auto conv_fn = [&]() {
      for (int64_t i = 0; i < out_h; ++i) {
        for (int64_t j = 0; j < out_w; ++j) {
          double acc = 0.0;
          for (int64_t ki = 0; ki < kh; ++ki) {
            for (int64_t kj = 0; kj < kw; ++kj) {
              acc +=
                  input[static_cast<size_t>((i + ki) * w + (j + kj))] *
                  kernel[static_cast<size_t>(ki * kw + kj)];
            }
          }
          out_conv[static_cast<size_t>(i * out_w + j)] = acc;
        }
      }
      sink += out_conv[0];
      return true;
    };
    const double conv_ns = MeasureNs(opts.warmup, opts.iters, conv_fn);
    if (conv_ns > 0) {
      const double bytes = elem_bytes * static_cast<double>(
                                            h * w + kh * kw + out_h * out_w);
      const double flops =
          2.0 * static_cast<double>(out_h * out_w * kh * kw);
      const double sec = conv_ns * 1e-9;
      AppendResult(results, "conv2d", "128x128_k3", "latency_us", "us",
                   conv_ns / 1000.0, backend);
      AppendResult(results, "conv2d", "128x128_k3", "bandwidth_gbps", "GB/s",
                   (bytes / sec) / 1e9, backend);
      AppendResult(results, "conv2d", "128x128_k3", "gflops", "GFLOP/s",
                   (flops / sec) / 1e9, backend);
    }
  }

  (void)sink;
}

void RunTuning(const BenchOptions& opts,
               std::vector<BenchResult>* results,
               const lattice::runtime::Backend* backend) {
  using lattice::runtime::BackendType;
  const size_t n = 1 << 20;
  const double elem_bytes = sizeof(float);
  const double bytes = 3.0 * elem_bytes * static_cast<double>(n);
  const std::vector<uint32_t> candidates = {64, 128, 256, 512};

  if (backend->Type() == BackendType::kCUDA) {
    const auto* cuda = static_cast<const lattice::runtime::CudaBackend*>(backend);
    auto kernel_or =
        cuda->BuildKernelFromFile("lattice_smoke.cu", "vec_mul", "");
    if (!kernel_or.ok()) {
      return;
    }
    auto kernel = kernel_or.value();
    auto buf_a_or = cuda->CreateBuffer(0, n * sizeof(float));
    auto buf_b_or = cuda->CreateBuffer(0, n * sizeof(float));
    auto buf_out_or = cuda->CreateBuffer(0, n * sizeof(float));
    if (!buf_a_or.ok() || !buf_b_or.ok() || !buf_out_or.ok()) {
      return;
    }
    auto buf_a = buf_a_or.value();
    auto buf_b = buf_b_or.value();
    auto buf_out = buf_out_or.value();
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    cuda->WriteBuffer(0, buf_a, a.data(), a.size() * sizeof(float));
    cuda->WriteBuffer(0, buf_b, b.data(), b.size() * sizeof(float));

    double best_gbps = 0.0;
    uint32_t best_block = 0;
    for (uint32_t block : candidates) {
      lattice::runtime::CudaLaunchConfig cfg;
      cfg.block[0] = block;
      cfg.grid[0] = static_cast<uint32_t>((n + block - 1) / block);
      unsigned int count = static_cast<unsigned int>(n);
      std::vector<lattice::runtime::CudaKernelArg> args;
      args.push_back(lattice::runtime::CudaKernelArg::Device(buf_a.ptr));
      args.push_back(lattice::runtime::CudaKernelArg::Device(buf_b.ptr));
      args.push_back(lattice::runtime::CudaKernelArg::Device(buf_out.ptr));
      args.push_back(lattice::runtime::CudaKernelArg::Value(&count, sizeof(count)));
      auto ns = MeasureNs(opts.warmup, opts.iters, [&]() {
        return cuda->LaunchKernel(kernel, cfg, args).ok();
      });
      if (ns > 0) {
        double sec = ns * 1e-9;
        double gbps = (bytes / sec) / 1e9;
        if (gbps > best_gbps) {
          best_gbps = gbps;
          best_block = block;
        }
      }
    }
    if (best_block != 0) {
      AppendResult(results, "tune_vec_mul", "n=1048576", "best_block", "threads",
                   static_cast<double>(best_block), backend);
      AppendResult(results, "tune_vec_mul", "n=1048576", "bandwidth_gbps", "GB/s",
                   best_gbps, backend);
    }
    cuda->ReleaseBuffer(&buf_a);
    cuda->ReleaseBuffer(&buf_b);
    cuda->ReleaseBuffer(&buf_out);
    cuda->ReleaseKernel(&kernel);
  } else if (backend->Type() == BackendType::kHIP) {
    const auto* hip = static_cast<const lattice::runtime::HipBackend*>(backend);
    auto kernel_or = hip->BuildKernelFromFile("lattice_smoke.hip", "vec_mul", "");
    if (!kernel_or.ok()) {
      return;
    }
    auto kernel = kernel_or.value();
    auto buf_a_or = hip->CreateBuffer(0, n * sizeof(float));
    auto buf_b_or = hip->CreateBuffer(0, n * sizeof(float));
    auto buf_out_or = hip->CreateBuffer(0, n * sizeof(float));
    if (!buf_a_or.ok() || !buf_b_or.ok() || !buf_out_or.ok()) {
      return;
    }
    auto buf_a = buf_a_or.value();
    auto buf_b = buf_b_or.value();
    auto buf_out = buf_out_or.value();
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    hip->WriteBuffer(0, buf_a, a.data(), a.size() * sizeof(float));
    hip->WriteBuffer(0, buf_b, b.data(), b.size() * sizeof(float));

    double best_gbps = 0.0;
    uint32_t best_block = 0;
    for (uint32_t block : candidates) {
      lattice::runtime::HipLaunchConfig cfg;
      cfg.block[0] = block;
      cfg.grid[0] = static_cast<uint32_t>((n + block - 1) / block);
      unsigned int count = static_cast<unsigned int>(n);
      std::vector<lattice::runtime::HipKernelArg> args;
      args.push_back(lattice::runtime::HipKernelArg::Device(buf_a.ptr));
      args.push_back(lattice::runtime::HipKernelArg::Device(buf_b.ptr));
      args.push_back(lattice::runtime::HipKernelArg::Device(buf_out.ptr));
      args.push_back(lattice::runtime::HipKernelArg::Value(&count, sizeof(count)));
      auto ns = MeasureNs(opts.warmup, opts.iters, [&]() {
        return hip->LaunchKernel(kernel, cfg, args).ok();
      });
      if (ns > 0) {
        double sec = ns * 1e-9;
        double gbps = (bytes / sec) / 1e9;
        if (gbps > best_gbps) {
          best_gbps = gbps;
          best_block = block;
        }
      }
    }
    if (best_block != 0) {
      AppendResult(results, "tune_vec_mul", "n=1048576", "best_block", "threads",
                   static_cast<double>(best_block), backend);
      AppendResult(results, "tune_vec_mul", "n=1048576", "bandwidth_gbps", "GB/s",
                   best_gbps, backend);
    }
    hip->ReleaseBuffer(&buf_a);
    hip->ReleaseBuffer(&buf_b);
    hip->ReleaseBuffer(&buf_out);
    hip->ReleaseKernel(&kernel);
  } else if (backend->Type() == BackendType::kOpenCL) {
    const auto* opencl =
        static_cast<const lattice::runtime::OpenCLBackend*>(backend);
    auto kernel_or = opencl->BuildKernelFromFile("lattice_smoke.cl", "vec_mul", "");
    if (!kernel_or.ok()) {
      return;
    }
    auto kernel = kernel_or.value();
    auto buf_a_or =
        opencl->CreateBuffer(0, n * sizeof(float), CL_MEM_READ_ONLY);
    auto buf_b_or =
        opencl->CreateBuffer(0, n * sizeof(float), CL_MEM_READ_ONLY);
    auto buf_out_or =
        opencl->CreateBuffer(0, n * sizeof(float), CL_MEM_WRITE_ONLY);
    if (!buf_a_or.ok() || !buf_b_or.ok() || !buf_out_or.ok()) {
      return;
    }
    auto buf_a = buf_a_or.value();
    auto buf_b = buf_b_or.value();
    auto buf_out = buf_out_or.value();
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    opencl->WriteBuffer(0, buf_a, a.data(), a.size() * sizeof(float));
    opencl->WriteBuffer(0, buf_b, b.data(), b.size() * sizeof(float));

    double best_gbps = 0.0;
    uint32_t best_block = 0;
    for (uint32_t block : candidates) {
      lattice::runtime::OpenCLLaunchConfig cfg;
      cfg.dims = 1;
      cfg.use_local = true;
      cfg.local[0] = block;
      const size_t global = ((n + block - 1) / block) * block;
      cfg.global[0] = global;
      const cl_uint count = static_cast<cl_uint>(n);
      std::vector<lattice::runtime::OpenCLKernelArg> args;
      args.push_back(lattice::runtime::OpenCLKernelArg::Mem(buf_a.mem));
      args.push_back(lattice::runtime::OpenCLKernelArg::Mem(buf_b.mem));
      args.push_back(lattice::runtime::OpenCLKernelArg::Mem(buf_out.mem));
      args.push_back(lattice::runtime::OpenCLKernelArg::Value(&count, sizeof(count)));
      auto ns = MeasureNs(opts.warmup, opts.iters, [&]() {
        return opencl->LaunchKernel(kernel, cfg, args).ok();
      });
      if (ns > 0) {
        double sec = ns * 1e-9;
        double gbps = (bytes / sec) / 1e9;
        if (gbps > best_gbps) {
          best_gbps = gbps;
          best_block = block;
        }
      }
    }
    if (best_block != 0) {
      AppendResult(results, "tune_vec_mul", "n=1048576", "best_block", "threads",
                   static_cast<double>(best_block), backend);
      AppendResult(results, "tune_vec_mul", "n=1048576", "bandwidth_gbps", "GB/s",
                   best_gbps, backend);
    }
    opencl->ReleaseBuffer(&buf_a);
    opencl->ReleaseBuffer(&buf_b);
    opencl->ReleaseBuffer(&buf_out);
    opencl->ReleaseKernel(&kernel);
  }
#if defined(__APPLE__)
  else if (backend->Type() == BackendType::kMetal) {
    const auto* metal =
        static_cast<const lattice::runtime::MetalBackend*>(backend);
    auto kernel_or =
        metal->BuildKernelFromFile("lattice_smoke.metal", "vec_mul", "");
    if (!kernel_or.ok()) {
      return;
    }
    auto kernel = kernel_or.value();
    auto buf_a_or = metal->CreateBuffer(0, n * sizeof(float));
    auto buf_b_or = metal->CreateBuffer(0, n * sizeof(float));
    auto buf_out_or = metal->CreateBuffer(0, n * sizeof(float));
    if (!buf_a_or.ok() || !buf_b_or.ok() || !buf_out_or.ok()) {
      return;
    }
    auto buf_a = buf_a_or.value();
    auto buf_b = buf_b_or.value();
    auto buf_out = buf_out_or.value();
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    metal->WriteBuffer(0, buf_a, a.data(), a.size() * sizeof(float));
    metal->WriteBuffer(0, buf_b, b.data(), b.size() * sizeof(float));

    double best_gbps = 0.0;
    uint32_t best_block = 0;
    for (uint32_t block : candidates) {
      lattice::runtime::MetalLaunchConfig cfg;
      cfg.grid[0] = static_cast<uint32_t>(n);
      cfg.threads[0] = block;
      cfg.use_threads = true;
      unsigned int count = static_cast<unsigned int>(n);
      std::vector<lattice::runtime::MetalKernelArg> args;
      args.push_back(lattice::runtime::MetalKernelArg::Buffer(buf_a.handle));
      args.push_back(lattice::runtime::MetalKernelArg::Buffer(buf_b.handle));
      args.push_back(lattice::runtime::MetalKernelArg::Buffer(buf_out.handle));
      args.push_back(lattice::runtime::MetalKernelArg::Value(&count, sizeof(count)));
      auto ns = MeasureNs(opts.warmup, opts.iters, [&]() {
        return metal->LaunchKernel(kernel, cfg, args).ok();
      });
      if (ns > 0) {
        double sec = ns * 1e-9;
        double gbps = (bytes / sec) / 1e9;
        if (gbps > best_gbps) {
          best_gbps = gbps;
          best_block = block;
        }
      }
    }
    if (best_block != 0) {
      AppendResult(results, "tune_vec_mul", "n=1048576", "best_block", "threads",
                   static_cast<double>(best_block), backend);
      AppendResult(results, "tune_vec_mul", "n=1048576", "bandwidth_gbps", "GB/s",
                   best_gbps, backend);
    }
    metal->ReleaseBuffer(&buf_a);
    metal->ReleaseBuffer(&buf_b);
    metal->ReleaseBuffer(&buf_out);
    metal->ReleaseKernel(&kernel);
  }
#else
  (void)opts;
  (void)results;
#endif
}

int Main(int argc, char** argv) {
  BenchOptions opts = ParseArgs(argc, argv);
  auto* backend = lattice::runtime::GetDefaultBackend();
  if (!backend) {
    std::cerr << "No backend selected\n";
    return 1;
  }

  std::vector<BenchResult> results;
  if (backend->Type() == lattice::runtime::BackendType::kCPU) {
    RunCpuMicrobench(opts, &results, backend);
  } else {
    auto* mutable_backend = const_cast<lattice::runtime::Backend*>(backend);
    lattice::runtime::ExecutionConfig config =
        mutable_backend->GetExecutionConfig();
    config.sync_on_launch = true;
    mutable_backend->SetExecutionConfig(config);
    RunMicrobench(opts, &results, backend);
    if (opts.tune) {
      RunTuning(opts, &results, backend);
    }
  }

  std::ostream* out = &std::cout;
  std::ofstream file;
  if (!opts.output_path.empty()) {
    file.open(opts.output_path);
    if (file) {
      out = &file;
    }
  }

  for (const auto& result : results) {
    *out << ResultLine(result) << "\n";
  }

  if (opts.check_baseline) {
    bool regression = false;
    auto baselines = LoadBaselines(opts.baseline_path);
    for (const auto& result : results) {
      auto entry = FindBaseline(baselines, result);
      if (!entry) {
        continue;
      }
      if (RegressionDetected(result, *entry)) {
        regression = true;
        std::cerr << "Regression: " << ResultLine(result)
                  << " baseline=" << entry->baseline
                  << " tol=" << entry->tolerance << "\n";
      }
    }
    if (regression) {
      return 2;
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  return Main(argc, argv);
}
