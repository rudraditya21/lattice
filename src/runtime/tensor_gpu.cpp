#include "runtime/tensor_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "runtime/backend.h"
#include "runtime/backends/backend_error.h"
#include "runtime/backends/backend_log.h"
#include "runtime/backends/cuda_abi.h"
#include "runtime/backends/cuda_backend.h"
#include "runtime/backends/device_caps.h"
#include "runtime/backends/hip_backend.h"
#include "runtime/backends/opencl_backend.h"
#include "runtime/ops.h"
#include "runtime/tensor_utils.h"
#include "util/error.h"

#if defined(__APPLE__)
#include "runtime/backends/metal_backend.h"
#endif

namespace lattice::runtime {

namespace {

constexpr size_t kMaxTensorDims = cuda::kMaxTensorDims;

using ElemwiseParams = cuda::ElemwiseParams;
using ReduceParams = cuda::ReduceParams;
using MatmulParams = cuda::MatmulParams;
using TransposeParams = cuda::TransposeParams;
using Conv2dParams = cuda::Conv2dParams;
using Pool2dParams = cuda::Pool2dParams;
using FftParams = cuda::FftParams;
using SolveParams = cuda::SolveParams;
using LuParams = cuda::LuParams;
using QrParams = cuda::QrParams;
using SvdParams = cuda::SvdParams;
using QuantileParams = cuda::QuantileParams;
using CorrelationParams = cuda::CorrelationParams;
using RegressionParams = cuda::RegressionParams;

struct KernelSpec {
  const char* name = nullptr;
  const char* file = nullptr;
};

KernelSpec ElemwiseKernel(parser::BinaryOp op) {
  switch (op) {
    case parser::BinaryOp::kAdd:
      return {"lattice_elemwise_add", "tensor_elemwise_add"};
    case parser::BinaryOp::kSub:
      return {"lattice_elemwise_sub", "tensor_elemwise_sub"};
    case parser::BinaryOp::kMul:
      return {"lattice_elemwise_mul", "tensor_elemwise_mul"};
    case parser::BinaryOp::kDiv:
      return {"lattice_elemwise_div", "tensor_elemwise_div"};
    default:
      return {};
  }
}

KernelSpec ReduceKernel(ReduceKind kind) {
  switch (kind) {
    case ReduceKind::kSum:
      return {"lattice_reduce_sum", "tensor_reduce_sum"};
    case ReduceKind::kMean:
      return {"lattice_reduce_mean", "tensor_reduce_mean"};
    case ReduceKind::kVar:
      return {"lattice_reduce_var", "tensor_reduce_var"};
    case ReduceKind::kStd:
      return {"lattice_reduce_std", "tensor_reduce_std"};
  }
  return {};
}

struct LaunchConfig {
  uint32_t dims = 1;
  uint32_t grid[3] = {1, 1, 1};
  uint32_t block[3] = {1, 1, 1};
  bool use_local = false;
};

LaunchConfig Make1DLaunch(uint64_t count) {
  LaunchConfig cfg;
  cfg.dims = 1;
  cfg.block[0] = 256;
  cfg.grid[0] = static_cast<uint32_t>((count + cfg.block[0] - 1) / cfg.block[0]);
  cfg.use_local = true;
  return cfg;
}

LaunchConfig Make2DLaunch(uint64_t width, uint64_t height) {
  LaunchConfig cfg;
  cfg.dims = 2;
  cfg.block[0] = 16;
  cfg.block[1] = 16;
  cfg.grid[0] = static_cast<uint32_t>((width + cfg.block[0] - 1) / cfg.block[0]);
  cfg.grid[1] = static_cast<uint32_t>((height + cfg.block[1] - 1) / cfg.block[1]);
  cfg.use_local = true;
  return cfg;
}

LaunchConfig MakeSingleLaunch() {
  LaunchConfig cfg;
  cfg.dims = 1;
  cfg.block[0] = 1;
  cfg.grid[0] = 1;
  cfg.use_local = false;
  return cfg;
}

struct GpuBuffer {
  BackendType backend = BackendType::kCPU;
  int device_index = 0;
  size_t bytes = 0;
  std::variant<OpenCLBuffer, CudaBuffer, HipBuffer, MetalBuffer> handle;
};

struct GpuKernel {
  BackendType backend = BackendType::kCPU;
  int device_index = 0;
  std::string name;
  std::variant<OpenCLKernel, CudaKernel, HipKernel, MetalKernel> handle;
};

struct GpuArg {
  enum class Kind { kBuffer, kValue };
  Kind kind = Kind::kValue;
  const GpuBuffer* buffer = nullptr;
  std::vector<uint8_t> value;

  static GpuArg Buffer(const GpuBuffer& buffer) {
    GpuArg arg;
    arg.kind = Kind::kBuffer;
    arg.buffer = &buffer;
    return arg;
  }

  static GpuArg Value(const void* data, size_t size) {
    GpuArg arg;
    arg.kind = Kind::kValue;
    arg.value.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
    return arg;
  }
};

struct BufferCleanup {
  explicit BufferCleanup(class GpuExecutor& exec) : exec_(exec) {}
  ~BufferCleanup();
  void Add(GpuBuffer* buf) { buffers_.push_back(buf); }

 private:
  class GpuExecutor& exec_;
  std::vector<GpuBuffer*> buffers_;
};

class GpuExecutor {
 public:
  static GpuExecutor& Instance() {
    static GpuExecutor exec;
    return exec;
  }

  Status EnsureInitialized() {
    std::lock_guard<std::mutex> lock(mu_);
    if (initialized_) return init_status_;
    initialized_ = true;

    const Backend* backend = GetDefaultBackend();
    if (!backend || backend->Type() == BackendType::kCPU) {
      init_status_ = Status::Unavailable("GPU backend not selected");
      return init_status_;
    }

    auto stream_or = backend->CreateStream();
    if (!stream_or.ok()) {
      init_status_ = stream_or.status();
      return init_status_;
    }

    backend_type_ = backend->Type();
    switch (backend_type_) {
      case BackendType::kOpenCL:
        opencl_ = static_cast<const OpenCLBackend*>(backend);
        device_count_ = opencl_->DeviceCount();
        break;
      case BackendType::kCUDA:
        cuda_ = static_cast<const CudaBackend*>(backend);
        device_count_ = cuda_->DeviceCount();
        break;
      case BackendType::kHIP:
        hip_ = static_cast<const HipBackend*>(backend);
        device_count_ = hip_->DeviceCount();
        break;
      case BackendType::kMetal:
#if defined(__APPLE__)
        metal_ = static_cast<const MetalBackend*>(backend);
        device_count_ = metal_->DeviceCount();
        break;
#else
        init_status_ = Status::Unavailable("Metal backend unavailable on this platform");
        return init_status_;
#endif
      case BackendType::kCPU:
        init_status_ = Status::Unavailable("GPU backend not selected");
        return init_status_;
    }

    if (device_count_ <= 0) {
      init_status_ = Status::Unavailable("No GPU devices available");
      return init_status_;
    }

    device_index_ = 0;
    use_fp64_ = false;
    std::vector<DeviceCapabilities> caps;
    switch (backend_type_) {
      case BackendType::kOpenCL:
        caps = opencl_->DeviceCaps();
        break;
      case BackendType::kCUDA:
        caps = cuda_->DeviceCaps();
        break;
      case BackendType::kHIP:
        caps = hip_->DeviceCaps();
        break;
      case BackendType::kMetal:
#if defined(__APPLE__)
        caps = metal_->DeviceCaps();
#endif
        break;
      case BackendType::kCPU:
        break;
    }
    if (!caps.empty() && caps[0].fp64 == CapabilityStatus::kYes) {
      use_fp64_ = true;
    }

    kernel_dir_ = FindKernelDir(backend_type_);
    if (kernel_dir_.empty()) {
      init_status_ = Status::Unavailable("Kernel directory not found");
      return init_status_;
    }

    init_status_ = Status::OK();
    return init_status_;
  }

  bool Enabled() { return EnsureInitialized().ok(); }

  BackendType Type() const { return backend_type_; }
  int DeviceIndex() const { return device_index_; }
  bool UseFp64() const { return use_fp64_; }

  StatusOr<GpuBuffer> AllocateBuffer(size_t bytes) {
    Status status = EnsureInitialized();
    if (!status.ok()) return status;
    switch (backend_type_) {
      case BackendType::kOpenCL: {
        auto buf_or = opencl_->CreateBuffer(device_index_, bytes);
        if (!buf_or.ok()) return buf_or.status();
        GpuBuffer out;
        out.backend = backend_type_;
        out.device_index = device_index_;
        out.bytes = buf_or.value().bytes;
        out.handle = buf_or.value();
        return out;
      }
      case BackendType::kCUDA: {
        auto buf_or = cuda_->CreateBuffer(device_index_, bytes);
        if (!buf_or.ok()) return buf_or.status();
        GpuBuffer out;
        out.backend = backend_type_;
        out.device_index = device_index_;
        out.bytes = buf_or.value().bytes;
        out.handle = buf_or.value();
        return out;
      }
      case BackendType::kHIP: {
        auto buf_or = hip_->CreateBuffer(device_index_, bytes);
        if (!buf_or.ok()) return buf_or.status();
        GpuBuffer out;
        out.backend = backend_type_;
        out.device_index = device_index_;
        out.bytes = buf_or.value().bytes;
        out.handle = buf_or.value();
        return out;
      }
      case BackendType::kMetal: {
#if defined(__APPLE__)
        auto buf_or = metal_->CreateBuffer(device_index_, bytes);
        if (!buf_or.ok()) return buf_or.status();
        GpuBuffer out;
        out.backend = backend_type_;
        out.device_index = device_index_;
        out.bytes = buf_or.value().bytes;
        out.handle = buf_or.value();
        return out;
#else
        return Status::Unavailable("Metal backend unavailable on this platform");
#endif
      }
      case BackendType::kCPU:
        return Status::Unavailable("GPU backend not selected");
    }
    return Status::Unavailable("Unsupported backend");
  }

  Status ReleaseBuffer(GpuBuffer* buffer) {
    if (!buffer) return Status::OK();
    switch (buffer->backend) {
      case BackendType::kOpenCL: {
        auto buf = std::get<OpenCLBuffer>(buffer->handle);
        return opencl_->ReleaseBuffer(&buf);
      }
      case BackendType::kCUDA: {
        auto buf = std::get<CudaBuffer>(buffer->handle);
        return cuda_->ReleaseBuffer(&buf);
      }
      case BackendType::kHIP: {
        auto buf = std::get<HipBuffer>(buffer->handle);
        return hip_->ReleaseBuffer(&buf);
      }
      case BackendType::kMetal: {
#if defined(__APPLE__)
        auto buf = std::get<MetalBuffer>(buffer->handle);
        return metal_->ReleaseBuffer(&buf);
#else
        return Status::Unavailable("Metal backend unavailable on this platform");
#endif
      }
      case BackendType::kCPU:
        return Status::OK();
    }
    return Status::OK();
  }

  Status WriteBuffer(const GpuBuffer& buffer, const void* data, size_t bytes) {
    switch (buffer.backend) {
      case BackendType::kOpenCL:
        return opencl_->WriteBuffer(buffer.device_index, std::get<OpenCLBuffer>(buffer.handle),
                                    data, bytes);
      case BackendType::kCUDA:
        return cuda_->WriteBuffer(buffer.device_index, std::get<CudaBuffer>(buffer.handle), data,
                                  bytes);
      case BackendType::kHIP:
        return hip_->WriteBuffer(buffer.device_index, std::get<HipBuffer>(buffer.handle), data,
                                 bytes);
      case BackendType::kMetal:
#if defined(__APPLE__)
        return metal_->WriteBuffer(buffer.device_index, std::get<MetalBuffer>(buffer.handle), data,
                                   bytes);
#else
        return Status::Unavailable("Metal backend unavailable on this platform");
#endif
      case BackendType::kCPU:
        return Status::Unavailable("GPU backend not selected");
    }
    return Status::Unavailable("Unsupported backend");
  }

  Status ReadBuffer(const GpuBuffer& buffer, void* data, size_t bytes) {
    switch (buffer.backend) {
      case BackendType::kOpenCL:
        return opencl_->ReadBuffer(buffer.device_index, std::get<OpenCLBuffer>(buffer.handle), data,
                                   bytes);
      case BackendType::kCUDA:
        return cuda_->ReadBuffer(buffer.device_index, std::get<CudaBuffer>(buffer.handle), data,
                                 bytes);
      case BackendType::kHIP:
        return hip_->ReadBuffer(buffer.device_index, std::get<HipBuffer>(buffer.handle), data,
                                bytes);
      case BackendType::kMetal:
#if defined(__APPLE__)
        return metal_->ReadBuffer(buffer.device_index, std::get<MetalBuffer>(buffer.handle), data,
                                  bytes);
#else
        return Status::Unavailable("Metal backend unavailable on this platform");
#endif
      case BackendType::kCPU:
        return Status::Unavailable("GPU backend not selected");
    }
    return Status::Unavailable("Unsupported backend");
  }

  Status WriteBufferFromDouble(const GpuBuffer& buffer, const double* data, size_t count) {
    if (use_fp64_) {
      return WriteBuffer(buffer, data, count * sizeof(double));
    }
    std::vector<float> tmp(count);
    for (size_t i = 0; i < count; ++i) tmp[i] = static_cast<float>(data[i]);
    return WriteBuffer(buffer, tmp.data(), tmp.size() * sizeof(float));
  }

  StatusOr<std::vector<double>> ReadBufferToDouble(const GpuBuffer& buffer, size_t count) {
    std::vector<double> out(count, 0.0);
    if (use_fp64_) {
      Status status = ReadBuffer(buffer, out.data(), count * sizeof(double));
      if (!status.ok()) return status;
      return out;
    }
    std::vector<float> tmp(count, 0.0f);
    Status status = ReadBuffer(buffer, tmp.data(), count * sizeof(float));
    if (!status.ok()) return status;
    for (size_t i = 0; i < count; ++i) out[i] = static_cast<double>(tmp[i]);
    return out;
  }

  StatusOr<GpuKernel> GetKernel(const KernelSpec& spec) {
    Status status = EnsureInitialized();
    if (!status.ok()) return status;
    if (!spec.name || !spec.file) {
      return Status::Invalid("Kernel spec is missing");
    }
    const std::string key = KernelCacheKey(spec.name);
    {
      std::lock_guard<std::mutex> lock(cache_mu_);
      auto it = kernel_cache_.find(key);
      if (it != kernel_cache_.end()) return it->second;
    }

    const std::string path = KernelPath(spec.file);
    if (path.empty()) {
      return Status::Unavailable("Kernel path not found");
    }

    GpuKernel kernel;
    switch (backend_type_) {
      case BackendType::kOpenCL: {
        auto kernels_or = opencl_->BuildKernelsFromFile(path, spec.name);
        if (!kernels_or.ok()) return kernels_or.status();
        kernel = WrapKernel(kernels_or.value());
        break;
      }
      case BackendType::kCUDA: {
        auto kernels_or = cuda_->BuildKernelsFromFile(path, spec.name);
        if (!kernels_or.ok()) return kernels_or.status();
        kernel = WrapKernel(kernels_or.value());
        break;
      }
      case BackendType::kHIP: {
        auto kernels_or = hip_->BuildKernelsFromFile(path, spec.name);
        if (!kernels_or.ok()) return kernels_or.status();
        kernel = WrapKernel(kernels_or.value());
        break;
      }
      case BackendType::kMetal: {
#if defined(__APPLE__)
        auto kernels_or = metal_->BuildKernelsFromFile(path, spec.name);
        if (!kernels_or.ok()) return kernels_or.status();
        kernel = WrapKernel(kernels_or.value());
        break;
#else
        return Status::Unavailable("Metal backend unavailable on this platform");
#endif
      }
      case BackendType::kCPU:
        return Status::Unavailable("GPU backend not selected");
    }

    kernel.backend = backend_type_;
    kernel.name = spec.name;
    kernel.device_index = device_index_;

    {
      std::lock_guard<std::mutex> lock(cache_mu_);
      kernel_cache_[key] = kernel;
    }
    return kernel;
  }

  Status LaunchKernel(const GpuKernel& kernel, const LaunchConfig& cfg,
                      const std::vector<GpuArg>& args) {
    switch (kernel.backend) {
      case BackendType::kOpenCL: {
        std::vector<OpenCLKernelArg> oargs;
        oargs.reserve(args.size());
        for (const auto& arg : args) {
          if (arg.kind == GpuArg::Kind::kBuffer) {
            const auto& buf = std::get<OpenCLBuffer>(arg.buffer->handle);
            oargs.push_back(OpenCLKernelArg::Mem(buf.mem));
          } else {
            oargs.push_back(OpenCLKernelArg::Value(arg.value.data(), arg.value.size()));
          }
        }
        OpenCLLaunchConfig config;
        config.dims = cfg.dims;
        config.global[0] = cfg.grid[0] * cfg.block[0];
        config.global[1] = cfg.grid[1] * cfg.block[1];
        config.global[2] = cfg.grid[2] * cfg.block[2];
        config.local[0] = cfg.block[0];
        config.local[1] = cfg.block[1];
        config.local[2] = cfg.block[2];
        config.use_local = cfg.use_local;
        return opencl_->LaunchKernel(std::get<OpenCLKernel>(kernel.handle), config, oargs);
      }
      case BackendType::kCUDA: {
        std::vector<CudaKernelArg> cargs;
        cargs.reserve(args.size());
        for (const auto& arg : args) {
          if (arg.kind == GpuArg::Kind::kBuffer) {
            const auto& buf = std::get<CudaBuffer>(arg.buffer->handle);
            cargs.push_back(CudaKernelArg::Device(buf.ptr));
          } else {
            cargs.push_back(CudaKernelArg::Value(arg.value.data(), arg.value.size()));
          }
        }
        CudaLaunchConfig config;
        config.grid[0] = cfg.grid[0];
        config.grid[1] = cfg.grid[1];
        config.grid[2] = cfg.grid[2];
        config.block[0] = cfg.block[0];
        config.block[1] = cfg.block[1];
        config.block[2] = cfg.block[2];
        return cuda_->LaunchKernel(std::get<CudaKernel>(kernel.handle), config, cargs);
      }
      case BackendType::kHIP: {
        std::vector<HipKernelArg> hargs;
        hargs.reserve(args.size());
        for (const auto& arg : args) {
          if (arg.kind == GpuArg::Kind::kBuffer) {
            const auto& buf = std::get<HipBuffer>(arg.buffer->handle);
            hargs.push_back(HipKernelArg::Device(buf.ptr));
          } else {
            hargs.push_back(HipKernelArg::Value(arg.value.data(), arg.value.size()));
          }
        }
        HipLaunchConfig config;
        config.grid[0] = cfg.grid[0];
        config.grid[1] = cfg.grid[1];
        config.grid[2] = cfg.grid[2];
        config.block[0] = cfg.block[0];
        config.block[1] = cfg.block[1];
        config.block[2] = cfg.block[2];
        return hip_->LaunchKernel(std::get<HipKernel>(kernel.handle), config, hargs);
      }
      case BackendType::kMetal: {
#if defined(__APPLE__)
        std::vector<MetalKernelArg> margs;
        margs.reserve(args.size());
        for (const auto& arg : args) {
          if (arg.kind == GpuArg::Kind::kBuffer) {
            const auto& buf = std::get<MetalBuffer>(arg.buffer->handle);
            margs.push_back(MetalKernelArg::Buffer(buf.handle));
          } else {
            margs.push_back(MetalKernelArg::Value(arg.value.data(), arg.value.size()));
          }
        }
        MetalLaunchConfig config;
        config.grid[0] = cfg.grid[0] * cfg.block[0];
        config.grid[1] = cfg.grid[1] * cfg.block[1];
        config.grid[2] = cfg.grid[2] * cfg.block[2];
        config.threads[0] = cfg.block[0];
        config.threads[1] = cfg.block[1];
        config.threads[2] = cfg.block[2];
        config.use_threads = cfg.use_local;
        return metal_->LaunchKernel(std::get<MetalKernel>(kernel.handle), config, margs);
#else
        return Status::Unavailable("Metal backend unavailable on this platform");
#endif
      }
      case BackendType::kCPU:
        return Status::Unavailable("GPU backend not selected");
    }
    return Status::Unavailable("Unsupported backend");
  }

 private:
  GpuExecutor() = default;

  std::string KernelPath(const std::string& file_base) const {
    if (kernel_dir_.empty()) return "";
    std::string ext;
    switch (backend_type_) {
      case BackendType::kOpenCL:
        ext = ".cl";
        break;
      case BackendType::kCUDA:
        ext = ".cu";
        break;
      case BackendType::kHIP:
        ext = ".hip";
        break;
      case BackendType::kMetal:
        ext = ".metal";
        break;
      case BackendType::kCPU:
        return "";
    }
    return (std::filesystem::path(kernel_dir_) / (file_base + ext)).string();
  }

  std::string KernelCacheKey(const std::string& name) const {
    std::ostringstream key;
    key << BackendTypeName(backend_type_) << ":" << device_index_ << ":" << name;
    return key.str();
  }

  std::string FindKernelDir(BackendType backend) const {
    if (const char* env = std::getenv("LATTICE_KERNEL_DIR")) {
      return env;
    }
    std::string folder;
    switch (backend) {
      case BackendType::kOpenCL:
        folder = "OpenCL";
        break;
      case BackendType::kCUDA:
        folder = "CUDA";
        break;
      case BackendType::kHIP:
        folder = "HIP";
        break;
      case BackendType::kMetal:
        folder = "Metal";
        break;
      case BackendType::kCPU:
        return "";
    }
    std::filesystem::path cwd = std::filesystem::current_path();
    for (int depth = 0; depth < 4; ++depth) {
      std::filesystem::path candidate = cwd / folder;
      if (std::filesystem::exists(candidate)) return candidate.string();
      if (!cwd.has_parent_path()) break;
      cwd = cwd.parent_path();
    }
    return "";
  }

  template <typename T>
  GpuKernel WrapKernel(const std::vector<T>& kernels) const {
    for (const auto& kernel : kernels) {
      if (kernel.device_index == device_index_) {
        GpuKernel out;
        out.backend = backend_type_;
        out.device_index = device_index_;
        out.name = kernel.name;
        out.handle = kernel;
        return out;
      }
    }
    if (!kernels.empty()) {
      GpuKernel out;
      out.backend = backend_type_;
      out.device_index = device_index_;
      out.name = kernels.front().name;
      out.handle = kernels.front();
      return out;
    }
    return {};
  }

  BackendType backend_type_ = BackendType::kCPU;
  const OpenCLBackend* opencl_ = nullptr;
  const CudaBackend* cuda_ = nullptr;
  const HipBackend* hip_ = nullptr;
#if defined(__APPLE__)
  const MetalBackend* metal_ = nullptr;
#else
  const MetalBackend* metal_ = nullptr;
#endif
  int device_index_ = 0;
  int device_count_ = 0;
  bool use_fp64_ = false;
  bool initialized_ = false;
  Status init_status_ = Status::OK();
  std::string kernel_dir_;
  std::mutex mu_;
  std::mutex cache_mu_;
  std::unordered_map<std::string, GpuKernel> kernel_cache_;
};

BufferCleanup::~BufferCleanup() {
  for (auto* buf : buffers_) {
    exec_.ReleaseBuffer(buf);
  }
}

bool FillParamArray(uint64_t* dst, size_t dst_len, const std::vector<int64_t>& src) {
  if (dst_len == 0) return false;
  if (src.size() > dst_len) return false;
  for (size_t i = 0; i < dst_len; ++i) {
    dst[i] = i < src.size() ? static_cast<uint64_t>(src[i]) : 0;
  }
  return true;
}

void LogGpuError(BackendType backend, const Status& status, const std::string& op) {
  if (!BackendVerboseEnabled(backend)) return;
  LogBackend({LogLevel::kWarn, backend, BackendErrorKind::kRuntime, status.message, op});
}

std::optional<Value> BuildDenseTensor(const std::vector<int64_t>& shape, DType elem_type,
                                      const std::vector<double>& data) {
  Value out = Value::Tensor(shape, elem_type, 0.0);
  if (static_cast<size_t>(out.tensor.size) != data.size()) return std::nullopt;
  for (size_t i = 0; i < data.size(); ++i) {
    out.tensor.Data()[static_cast<int64_t>(i)] = data[i];
  }
  out.tensor.elem_type = elem_type;
  return out;
}

}  // namespace

std::optional<Value> TryGpuElemwise(const Value& lhs, const Value& rhs, parser::BinaryOp op,
                                    int line, int column, std::string* error) {
  KernelSpec spec = ElemwiseKernel(op);
  if (!spec.name) return std::nullopt;

  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;

  if ((lhs.type == DType::kTensor && lhs.tensor.kind == TensorKind::kRagged) ||
      (rhs.type == DType::kTensor && rhs.tensor.kind == TensorKind::kRagged)) {
    if (lhs.type != DType::kTensor || rhs.type != DType::kTensor ||
        lhs.tensor.kind != TensorKind::kRagged || rhs.tensor.kind != TensorKind::kRagged) {
      throw util::Error(
          "Ragged tensors only support ops with matching ragged tensors (use to_dense)", line,
          column);
    }
    if (lhs.tensor.row_splits != rhs.tensor.row_splits) {
      throw util::Error("Ragged tensors must share identical row_splits for elementwise ops", line,
                        column);
    }
    if (lhs.tensor.ragged_values.size() != rhs.tensor.ragged_values.size()) {
      throw util::Error("Ragged value buffers must match in length", line, column);
    }

    const size_t count = lhs.tensor.ragged_values.size();
    auto kernel_or = exec.GetKernel(spec);
    if (!kernel_or.ok()) {
      LogGpuError(exec.Type(), kernel_or.status(), "elemwise");
      if (error) *error = kernel_or.status().message;
      return std::nullopt;
    }

    BufferCleanup cleanup(exec);
    auto lhs_buf_or =
        exec.AllocateBuffer(count * (exec.UseFp64() ? sizeof(double) : sizeof(float)));
    if (!lhs_buf_or.ok()) {
      LogGpuError(exec.Type(), lhs_buf_or.status(), "elemwise");
      if (error) *error = lhs_buf_or.status().message;
      return std::nullopt;
    }
    auto rhs_buf_or =
        exec.AllocateBuffer(count * (exec.UseFp64() ? sizeof(double) : sizeof(float)));
    if (!rhs_buf_or.ok()) {
      LogGpuError(exec.Type(), rhs_buf_or.status(), "elemwise");
      if (error) *error = rhs_buf_or.status().message;
      return std::nullopt;
    }
    auto out_buf_or =
        exec.AllocateBuffer(count * (exec.UseFp64() ? sizeof(double) : sizeof(float)));
    if (!out_buf_or.ok()) {
      LogGpuError(exec.Type(), out_buf_or.status(), "elemwise");
      if (error) *error = out_buf_or.status().message;
      return std::nullopt;
    }

    GpuBuffer lhs_buf = lhs_buf_or.value();
    GpuBuffer rhs_buf = rhs_buf_or.value();
    GpuBuffer out_buf = out_buf_or.value();
    cleanup.Add(&lhs_buf);
    cleanup.Add(&rhs_buf);
    cleanup.Add(&out_buf);

    std::vector<double> lhs_vals(lhs.tensor.ragged_values.begin(), lhs.tensor.ragged_values.end());
    std::vector<double> rhs_vals(rhs.tensor.ragged_values.begin(), rhs.tensor.ragged_values.end());
    Status status = exec.WriteBufferFromDouble(lhs_buf, lhs_vals.data(), lhs_vals.size());
    if (!status.ok()) {
      LogGpuError(exec.Type(), status, "elemwise");
      if (error) *error = status.message;
      return std::nullopt;
    }
    status = exec.WriteBufferFromDouble(rhs_buf, rhs_vals.data(), rhs_vals.size());
    if (!status.ok()) {
      LogGpuError(exec.Type(), status, "elemwise");
      if (error) *error = status.message;
      return std::nullopt;
    }

    ElemwiseParams params;
    params.count = static_cast<uint64_t>(count);
    params.op = 0;
    params.dtype = 0;
    params.ndim = 1;
    params.flags = 0;
    params.shape[0] = static_cast<uint64_t>(count);
    params.out_strides[0] = 1;
    params.lhs_strides[0] = 1;
    params.rhs_strides[0] = 1;

    LaunchConfig cfg = Make1DLaunch(params.count);
    std::vector<GpuArg> args;
    args.push_back(GpuArg::Buffer(lhs_buf));
    args.push_back(GpuArg::Buffer(rhs_buf));
    args.push_back(GpuArg::Buffer(out_buf));
    args.push_back(GpuArg::Value(&params, sizeof(params)));

    status = exec.LaunchKernel(kernel_or.value(), cfg, args);
    if (!status.ok()) {
      LogGpuError(exec.Type(), status, "elemwise");
      if (error) *error = status.message;
      return std::nullopt;
    }

    auto out_data_or = exec.ReadBufferToDouble(out_buf, count);
    if (!out_data_or.ok()) {
      LogGpuError(exec.Type(), out_data_or.status(), "elemwise");
      if (error) *error = out_data_or.status().message;
      return std::nullopt;
    }

    Value out =
        Value::TensorRagged(lhs.tensor.row_splits, {},
                            PromoteType(lhs.tensor.elem_type, rhs.tensor.elem_type, line, column));
    out.tensor.ragged_values = std::move(out_data_or.value());
    return out;
  }

  Value lhs_d = lhs;
  Value rhs_d = rhs;
  bool lhs_sparse = lhs.type == DType::kTensor && lhs.tensor.kind != TensorKind::kDense;
  bool rhs_sparse = rhs.type == DType::kTensor && rhs.tensor.kind != TensorKind::kDense;
  bool sparse_out = false;
  TensorKind sparse_kind = TensorKind::kDense;

  if (lhs_sparse || rhs_sparse) {
    if (lhs.type == DType::kTensor && rhs.type == DType::kTensor &&
        lhs.tensor.kind == rhs.tensor.kind &&
        (lhs.tensor.kind == TensorKind::kSparseCSR || lhs.tensor.kind == TensorKind::kSparseCOO)) {
      if (lhs.tensor.shape != rhs.tensor.shape) {
        throw util::Error("Sparse tensors must share shape for elementwise ops", line, column);
      }
      sparse_out = true;
      sparse_kind = lhs.tensor.kind;
    } else if (lhs.type == DType::kTensor && rhs.type == DType::kTensor &&
               lhs.tensor.kind != rhs.tensor.kind && lhs.tensor.kind != TensorKind::kDense &&
               rhs.tensor.kind != TensorKind::kDense) {
      throw util::Error("Sparse tensor formats must match; convert first", line, column);
    }

    if (lhs.type == DType::kTensor && lhs.tensor.kind != TensorKind::kDense) {
      lhs_d = ToDenseTensor(lhs, line, column);
    }
    if (rhs.type == DType::kTensor && rhs.tensor.kind != TensorKind::kDense) {
      rhs_d = ToDenseTensor(rhs, line, column);
    }
  }

  std::vector<int64_t> lhs_shape =
      lhs_d.type == DType::kTensor ? lhs_d.tensor.shape : std::vector<int64_t>{};
  std::vector<int64_t> rhs_shape =
      rhs_d.type == DType::kTensor ? rhs_d.tensor.shape : std::vector<int64_t>{};
  auto broadcast_shape = BroadcastShape(lhs_shape, rhs_shape);
  if (!broadcast_shape.has_value()) {
    throw util::Error("Tensor shapes are not broadcastable (lhs " + ShapeToString(lhs_shape) +
                          ", rhs " + ShapeToString(rhs_shape) + ")",
                      0, 0);
  }

  const std::vector<int64_t>& out_shape = *broadcast_shape;
  if (out_shape.size() > kMaxTensorDims) {
    if (error) *error = "tensor rank exceeds GPU broadcast limit";
    return std::nullopt;
  }

  DType elem_target;
  if (lhs_d.type == DType::kTensor && rhs_d.type == DType::kTensor) {
    elem_target = PromoteType(lhs_d.tensor.elem_type, rhs_d.tensor.elem_type, line, column);
  } else if (lhs_d.type == DType::kTensor) {
    elem_target = PromoteType(lhs_d.tensor.elem_type, rhs_d.type, line, column);
  } else {
    elem_target = PromoteType(rhs_d.tensor.elem_type, lhs_d.type, line, column);
  }

  auto out_value = Value::Tensor(out_shape, elem_target, 0.0);

  std::vector<int64_t> lhs_bstrides =
      lhs_d.type == DType::kTensor
          ? BroadcastStrides(lhs_d.tensor.shape, lhs_d.tensor.strides, out_shape.size())
          : std::vector<int64_t>(out_shape.size(), 0);
  std::vector<int64_t> rhs_bstrides =
      rhs_d.type == DType::kTensor
          ? BroadcastStrides(rhs_d.tensor.shape, rhs_d.tensor.strides, out_shape.size())
          : std::vector<int64_t>(out_shape.size(), 0);

  if (lhs_bstrides.size() > kMaxTensorDims || rhs_bstrides.size() > kMaxTensorDims ||
      out_value.tensor.strides.size() > kMaxTensorDims) {
    if (error) *error = "tensor rank exceeds GPU broadcast limit";
    return std::nullopt;
  }

  auto kernel_or = exec.GetKernel(spec);
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "elemwise");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  BufferCleanup cleanup(exec);

  size_t lhs_count = lhs_d.type == DType::kTensor ? static_cast<size_t>(lhs_d.tensor.size) : 1;
  size_t rhs_count = rhs_d.type == DType::kTensor ? static_cast<size_t>(rhs_d.tensor.size) : 1;
  size_t out_count = static_cast<size_t>(out_value.tensor.size);
  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);

  auto lhs_buf_or = exec.AllocateBuffer(lhs_count * elem_bytes);
  if (!lhs_buf_or.ok()) {
    LogGpuError(exec.Type(), lhs_buf_or.status(), "elemwise");
    if (error) *error = lhs_buf_or.status().message;
    return std::nullopt;
  }
  auto rhs_buf_or = exec.AllocateBuffer(rhs_count * elem_bytes);
  if (!rhs_buf_or.ok()) {
    LogGpuError(exec.Type(), rhs_buf_or.status(), "elemwise");
    if (error) *error = rhs_buf_or.status().message;
    return std::nullopt;
  }
  auto out_buf_or = exec.AllocateBuffer(out_count * elem_bytes);
  if (!out_buf_or.ok()) {
    LogGpuError(exec.Type(), out_buf_or.status(), "elemwise");
    if (error) *error = out_buf_or.status().message;
    return std::nullopt;
  }

  GpuBuffer lhs_buf = lhs_buf_or.value();
  GpuBuffer rhs_buf = rhs_buf_or.value();
  GpuBuffer out_buf = out_buf_or.value();
  cleanup.Add(&lhs_buf);
  cleanup.Add(&rhs_buf);
  cleanup.Add(&out_buf);

  if (lhs_d.type == DType::kTensor) {
    Status status = exec.WriteBufferFromDouble(lhs_buf, lhs_d.tensor.Data(), lhs_count);
    if (!status.ok()) {
      LogGpuError(exec.Type(), status, "elemwise");
      if (error) *error = status.message;
      return std::nullopt;
    }
  } else {
    double scalar = lhs_d.f64;
    Status status = exec.WriteBufferFromDouble(lhs_buf, &scalar, 1);
    if (!status.ok()) {
      LogGpuError(exec.Type(), status, "elemwise");
      if (error) *error = status.message;
      return std::nullopt;
    }
  }

  if (rhs_d.type == DType::kTensor) {
    Status status = exec.WriteBufferFromDouble(rhs_buf, rhs_d.tensor.Data(), rhs_count);
    if (!status.ok()) {
      LogGpuError(exec.Type(), status, "elemwise");
      if (error) *error = status.message;
      return std::nullopt;
    }
  } else {
    double scalar = rhs_d.f64;
    Status status = exec.WriteBufferFromDouble(rhs_buf, &scalar, 1);
    if (!status.ok()) {
      LogGpuError(exec.Type(), status, "elemwise");
      if (error) *error = status.message;
      return std::nullopt;
    }
  }

  ElemwiseParams params;
  params.count = static_cast<uint64_t>(out_count);
  params.op = 0;
  params.dtype = 0;
  params.ndim = static_cast<uint32_t>(out_shape.size());
  params.flags = 0;
  FillParamArray(params.shape, kMaxTensorDims, out_shape);
  FillParamArray(params.out_strides, kMaxTensorDims, out_value.tensor.strides);
  FillParamArray(params.lhs_strides, kMaxTensorDims, lhs_bstrides);
  FillParamArray(params.rhs_strides, kMaxTensorDims, rhs_bstrides);

  LaunchConfig cfg = Make1DLaunch(params.count);
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(lhs_buf));
  args.push_back(GpuArg::Buffer(rhs_buf));
  args.push_back(GpuArg::Buffer(out_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  Status status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "elemwise");
    if (error) *error = status.message;
    return std::nullopt;
  }

  auto out_data_or = exec.ReadBufferToDouble(out_buf, out_count);
  if (!out_data_or.ok()) {
    LogGpuError(exec.Type(), out_data_or.status(), "elemwise");
    if (error) *error = out_data_or.status().message;
    return std::nullopt;
  }

  auto out_dense = BuildDenseTensor(out_shape, elem_target, out_data_or.value());
  if (!out_dense.has_value()) return std::nullopt;

  if (sparse_out) {
    return sparse_kind == TensorKind::kSparseCSR ? DenseToCSR(out_dense.value())
                                                 : DenseToCOO(out_dense.value());
  }
  return out_dense.value();
}

std::optional<Value> TryGpuReduce(const Value& v, ReduceKind kind, int line, int column,
                                  std::string* error) {
  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;
  if (v.type != DType::kTensor) return std::nullopt;
  if (v.tensor.kind == TensorKind::kRagged) return std::nullopt;

  Value dense = v;
  if (v.tensor.kind != TensorKind::kDense) {
    dense = ToDenseTensor(v, line, column);
  }

  KernelSpec spec = ReduceKernel(kind);
  if (!spec.name) return std::nullopt;
  auto kernel_or = exec.GetKernel(spec);
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "reduce");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  BufferCleanup cleanup(exec);
  size_t count = static_cast<size_t>(dense.tensor.size);
  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);

  auto in_buf_or = exec.AllocateBuffer(count * elem_bytes);
  if (!in_buf_or.ok()) {
    LogGpuError(exec.Type(), in_buf_or.status(), "reduce");
    if (error) *error = in_buf_or.status().message;
    return std::nullopt;
  }
  auto out_buf_or = exec.AllocateBuffer(elem_bytes);
  if (!out_buf_or.ok()) {
    LogGpuError(exec.Type(), out_buf_or.status(), "reduce");
    if (error) *error = out_buf_or.status().message;
    return std::nullopt;
  }

  GpuBuffer in_buf = in_buf_or.value();
  GpuBuffer out_buf = out_buf_or.value();
  cleanup.Add(&in_buf);
  cleanup.Add(&out_buf);

  Status status = exec.WriteBufferFromDouble(in_buf, dense.tensor.Data(), count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "reduce");
    if (error) *error = status.message;
    return std::nullopt;
  }

  ReduceParams params;
  params.count = static_cast<uint64_t>(count);
  params.op = 0;
  params.dtype = 0;
  params.stride = 0;

  LaunchConfig cfg = MakeSingleLaunch();
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(in_buf));
  args.push_back(GpuArg::Buffer(out_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "reduce");
    if (error) *error = status.message;
    return std::nullopt;
  }

  auto out_data_or = exec.ReadBufferToDouble(out_buf, 1);
  if (!out_data_or.ok()) {
    LogGpuError(exec.Type(), out_data_or.status(), "reduce");
    if (error) *error = out_data_or.status().message;
    return std::nullopt;
  }

  Value raw = Value::F64(out_data_or.value()[0]);
  return CastTo(dense.tensor.elem_type, raw, line, column);
}

std::optional<Value> TryGpuTranspose(const Value& v, int line, int column, std::string* error) {
  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;
  if (v.type != DType::kTensor) return std::nullopt;

  Value dense = ToDenseTensor(v, line, column);
  if (dense.tensor.shape.size() != 2) {
    throw util::Error("transpose supports only 2D tensors", line, column);
  }

  auto kernel_or = exec.GetKernel({"lattice_transpose", "tensor_transpose"});
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "transpose");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  const int64_t rows = dense.tensor.shape[0];
  const int64_t cols = dense.tensor.shape[1];
  const size_t count = static_cast<size_t>(dense.tensor.size);
  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);

  BufferCleanup cleanup(exec);
  auto in_buf_or = exec.AllocateBuffer(count * elem_bytes);
  if (!in_buf_or.ok()) {
    LogGpuError(exec.Type(), in_buf_or.status(), "transpose");
    if (error) *error = in_buf_or.status().message;
    return std::nullopt;
  }
  auto out_buf_or = exec.AllocateBuffer(count * elem_bytes);
  if (!out_buf_or.ok()) {
    LogGpuError(exec.Type(), out_buf_or.status(), "transpose");
    if (error) *error = out_buf_or.status().message;
    return std::nullopt;
  }

  GpuBuffer in_buf = in_buf_or.value();
  GpuBuffer out_buf = out_buf_or.value();
  cleanup.Add(&in_buf);
  cleanup.Add(&out_buf);

  Status status = exec.WriteBufferFromDouble(in_buf, dense.tensor.Data(), count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "transpose");
    if (error) *error = status.message;
    return std::nullopt;
  }

  TransposeParams params;
  params.rows = static_cast<uint64_t>(rows);
  params.cols = static_cast<uint64_t>(cols);

  LaunchConfig cfg = Make2DLaunch(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows));
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(in_buf));
  args.push_back(GpuArg::Buffer(out_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "transpose");
    if (error) *error = status.message;
    return std::nullopt;
  }

  auto out_data_or = exec.ReadBufferToDouble(out_buf, count);
  if (!out_data_or.ok()) {
    LogGpuError(exec.Type(), out_data_or.status(), "transpose");
    if (error) *error = out_data_or.status().message;
    return std::nullopt;
  }

  auto out_dense = BuildDenseTensor({cols, rows}, dense.tensor.elem_type, out_data_or.value());
  if (!out_dense.has_value()) return std::nullopt;
  return out_dense.value();
}

std::optional<Value> TryGpuMatmul(const Value& lhs, const Value& rhs, int line, int column,
                                  std::string* error) {
  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;
  if (lhs.type != DType::kTensor || rhs.type != DType::kTensor) {
    throw util::Error("matmul expects tensor arguments", line, column);
  }

  Value A = ToDenseTensor(lhs, line, column);
  Value B = ToDenseTensor(rhs, line, column);
  if (A.tensor.shape.size() != 2 || B.tensor.shape.size() != 2) {
    throw util::Error("matmul supports only 2D tensors", line, column);
  }
  int64_t m = A.tensor.shape[0];
  int64_t k = A.tensor.shape[1];
  int64_t k2 = B.tensor.shape[0];
  int64_t n = B.tensor.shape[1];
  if (k != k2) {
    throw util::Error("matmul shape mismatch", line, column);
  }

  auto kernel_or = exec.GetKernel({"lattice_matmul", "tensor_matmul"});
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "matmul");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);
  size_t a_count = static_cast<size_t>(A.tensor.size);
  size_t b_count = static_cast<size_t>(B.tensor.size);
  size_t c_count = static_cast<size_t>(m * n);

  BufferCleanup cleanup(exec);
  auto a_buf_or = exec.AllocateBuffer(a_count * elem_bytes);
  if (!a_buf_or.ok()) {
    LogGpuError(exec.Type(), a_buf_or.status(), "matmul");
    if (error) *error = a_buf_or.status().message;
    return std::nullopt;
  }
  auto b_buf_or = exec.AllocateBuffer(b_count * elem_bytes);
  if (!b_buf_or.ok()) {
    LogGpuError(exec.Type(), b_buf_or.status(), "matmul");
    if (error) *error = b_buf_or.status().message;
    return std::nullopt;
  }
  auto c_buf_or = exec.AllocateBuffer(c_count * elem_bytes);
  if (!c_buf_or.ok()) {
    LogGpuError(exec.Type(), c_buf_or.status(), "matmul");
    if (error) *error = c_buf_or.status().message;
    return std::nullopt;
  }

  GpuBuffer a_buf = a_buf_or.value();
  GpuBuffer b_buf = b_buf_or.value();
  GpuBuffer c_buf = c_buf_or.value();
  cleanup.Add(&a_buf);
  cleanup.Add(&b_buf);
  cleanup.Add(&c_buf);

  Status status = exec.WriteBufferFromDouble(a_buf, A.tensor.Data(), a_count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "matmul");
    if (error) *error = status.message;
    return std::nullopt;
  }
  status = exec.WriteBufferFromDouble(b_buf, B.tensor.Data(), b_count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "matmul");
    if (error) *error = status.message;
    return std::nullopt;
  }

  MatmulParams params;
  params.m = static_cast<uint64_t>(m);
  params.n = static_cast<uint64_t>(n);
  params.k = static_cast<uint64_t>(k);
  params.lda = static_cast<uint64_t>(k);
  params.ldb = static_cast<uint64_t>(n);
  params.ldc = static_cast<uint64_t>(n);
  params.dtype = 0;
  params.flags = 0;

  LaunchConfig cfg = Make2DLaunch(static_cast<uint64_t>(n), static_cast<uint64_t>(m));
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(a_buf));
  args.push_back(GpuArg::Buffer(b_buf));
  args.push_back(GpuArg::Buffer(c_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "matmul");
    if (error) *error = status.message;
    return std::nullopt;
  }

  auto out_data_or = exec.ReadBufferToDouble(c_buf, c_count);
  if (!out_data_or.ok()) {
    LogGpuError(exec.Type(), out_data_or.status(), "matmul");
    if (error) *error = out_data_or.status().message;
    return std::nullopt;
  }

  DType elem = PromoteType(A.tensor.elem_type, B.tensor.elem_type, line, column);
  auto out_dense = BuildDenseTensor({m, n}, elem, out_data_or.value());
  if (!out_dense.has_value()) return std::nullopt;
  return out_dense.value();
}

std::optional<Value> TryGpuConv2d(const Value& input, const Value& kernel, int line, int column,
                                  std::string* error) {
  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;
  Value in = ToDenseTensor(input, line, column);
  Value k = ToDenseTensor(kernel, line, column);
  if (in.type != DType::kTensor || k.type != DType::kTensor) {
    throw util::Error("conv2d expects tensor arguments", line, column);
  }
  if (in.tensor.shape.size() != 2 || k.tensor.shape.size() != 2) {
    throw util::Error("conv2d supports 2D tensors only", line, column);
  }
  int64_t h = in.tensor.shape[0];
  int64_t w = in.tensor.shape[1];
  int64_t kh = k.tensor.shape[0];
  int64_t kw = k.tensor.shape[1];
  if (kh > h || kw > w) {
    throw util::Error("conv2d kernel larger than input", line, column);
  }
  int64_t oh = h - kh + 1;
  int64_t ow = w - kw + 1;

  auto kernel_or = exec.GetKernel({"lattice_conv2d", "tensor_conv2d"});
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "conv2d");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);
  size_t in_count = static_cast<size_t>(in.tensor.size);
  size_t k_count = static_cast<size_t>(k.tensor.size);
  size_t out_count = static_cast<size_t>(oh * ow);

  BufferCleanup cleanup(exec);
  auto in_buf_or = exec.AllocateBuffer(in_count * elem_bytes);
  if (!in_buf_or.ok()) {
    LogGpuError(exec.Type(), in_buf_or.status(), "conv2d");
    if (error) *error = in_buf_or.status().message;
    return std::nullopt;
  }
  auto k_buf_or = exec.AllocateBuffer(k_count * elem_bytes);
  if (!k_buf_or.ok()) {
    LogGpuError(exec.Type(), k_buf_or.status(), "conv2d");
    if (error) *error = k_buf_or.status().message;
    return std::nullopt;
  }
  auto out_buf_or = exec.AllocateBuffer(out_count * elem_bytes);
  if (!out_buf_or.ok()) {
    LogGpuError(exec.Type(), out_buf_or.status(), "conv2d");
    if (error) *error = out_buf_or.status().message;
    return std::nullopt;
  }

  GpuBuffer in_buf = in_buf_or.value();
  GpuBuffer k_buf = k_buf_or.value();
  GpuBuffer out_buf = out_buf_or.value();
  cleanup.Add(&in_buf);
  cleanup.Add(&k_buf);
  cleanup.Add(&out_buf);

  Status status = exec.WriteBufferFromDouble(in_buf, in.tensor.Data(), in_count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "conv2d");
    if (error) *error = status.message;
    return std::nullopt;
  }
  status = exec.WriteBufferFromDouble(k_buf, k.tensor.Data(), k_count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "conv2d");
    if (error) *error = status.message;
    return std::nullopt;
  }

  Conv2dParams params;
  params.in_h = static_cast<uint64_t>(h);
  params.in_w = static_cast<uint64_t>(w);
  params.k_h = static_cast<uint64_t>(kh);
  params.k_w = static_cast<uint64_t>(kw);
  params.out_h = static_cast<uint64_t>(oh);
  params.out_w = static_cast<uint64_t>(ow);

  LaunchConfig cfg = Make2DLaunch(static_cast<uint64_t>(ow), static_cast<uint64_t>(oh));
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(in_buf));
  args.push_back(GpuArg::Buffer(k_buf));
  args.push_back(GpuArg::Buffer(out_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "conv2d");
    if (error) *error = status.message;
    return std::nullopt;
  }

  auto out_data_or = exec.ReadBufferToDouble(out_buf, out_count);
  if (!out_data_or.ok()) {
    LogGpuError(exec.Type(), out_data_or.status(), "conv2d");
    if (error) *error = out_data_or.status().message;
    return std::nullopt;
  }

  DType elem = PromoteType(in.tensor.elem_type, k.tensor.elem_type, line, column);
  auto out_dense = BuildDenseTensor({oh, ow}, elem, out_data_or.value());
  if (!out_dense.has_value()) return std::nullopt;
  return out_dense.value();
}

std::optional<Value> TryGpuMaxPool2d(const Value& input, int64_t k_h, int64_t k_w, int line,
                                     int column, std::string* error) {
  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;
  Value in = ToDenseTensor(input, line, column);
  if (in.type != DType::kTensor || in.tensor.shape.size() != 2) {
    throw util::Error("max_pool2d expects a 2D tensor", line, column);
  }
  if (k_h <= 0 || k_w <= 0) {
    throw util::Error("max_pool2d kernel sizes must be positive", line, column);
  }
  int64_t h = in.tensor.shape[0];
  int64_t w = in.tensor.shape[1];
  int64_t oh = h / k_h;
  int64_t ow = w / k_w;

  auto kernel_or = exec.GetKernel({"lattice_max_pool2d", "tensor_max_pool2d"});
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "max_pool2d");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);
  size_t in_count = static_cast<size_t>(in.tensor.size);
  size_t out_count = static_cast<size_t>(oh * ow);

  BufferCleanup cleanup(exec);
  auto in_buf_or = exec.AllocateBuffer(in_count * elem_bytes);
  if (!in_buf_or.ok()) {
    LogGpuError(exec.Type(), in_buf_or.status(), "max_pool2d");
    if (error) *error = in_buf_or.status().message;
    return std::nullopt;
  }
  auto out_buf_or = exec.AllocateBuffer(out_count * elem_bytes);
  if (!out_buf_or.ok()) {
    LogGpuError(exec.Type(), out_buf_or.status(), "max_pool2d");
    if (error) *error = out_buf_or.status().message;
    return std::nullopt;
  }

  GpuBuffer in_buf = in_buf_or.value();
  GpuBuffer out_buf = out_buf_or.value();
  cleanup.Add(&in_buf);
  cleanup.Add(&out_buf);

  Status status = exec.WriteBufferFromDouble(in_buf, in.tensor.Data(), in_count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "max_pool2d");
    if (error) *error = status.message;
    return std::nullopt;
  }

  Pool2dParams params;
  params.in_h = static_cast<uint64_t>(h);
  params.in_w = static_cast<uint64_t>(w);
  params.k_h = static_cast<uint64_t>(k_h);
  params.k_w = static_cast<uint64_t>(k_w);
  params.out_h = static_cast<uint64_t>(oh);
  params.out_w = static_cast<uint64_t>(ow);

  LaunchConfig cfg = Make2DLaunch(static_cast<uint64_t>(ow), static_cast<uint64_t>(oh));
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(in_buf));
  args.push_back(GpuArg::Buffer(out_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "max_pool2d");
    if (error) *error = status.message;
    return std::nullopt;
  }

  auto out_data_or = exec.ReadBufferToDouble(out_buf, out_count);
  if (!out_data_or.ok()) {
    LogGpuError(exec.Type(), out_data_or.status(), "max_pool2d");
    if (error) *error = out_data_or.status().message;
    return std::nullopt;
  }

  auto out_dense = BuildDenseTensor({oh, ow}, in.tensor.elem_type, out_data_or.value());
  if (!out_dense.has_value()) return std::nullopt;
  return out_dense.value();
}

std::optional<Value> TryGpuFft1d(const Value& input, int line, int column, std::string* error) {
  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;
  Value in = ToDenseTensor(input, line, column);
  if (in.type != DType::kTensor || in.tensor.shape.size() != 1) {
    throw util::Error("fft1d expects a 1D tensor", line, column);
  }
  int64_t n = in.tensor.shape[0];

  auto kernel_or = exec.GetKernel({"lattice_fft1d", "tensor_fft1d"});
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "fft1d");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);
  size_t count = static_cast<size_t>(n);

  BufferCleanup cleanup(exec);
  auto in_buf_or = exec.AllocateBuffer(count * elem_bytes);
  if (!in_buf_or.ok()) {
    LogGpuError(exec.Type(), in_buf_or.status(), "fft1d");
    if (error) *error = in_buf_or.status().message;
    return std::nullopt;
  }
  auto real_buf_or = exec.AllocateBuffer(count * elem_bytes);
  if (!real_buf_or.ok()) {
    LogGpuError(exec.Type(), real_buf_or.status(), "fft1d");
    if (error) *error = real_buf_or.status().message;
    return std::nullopt;
  }
  auto imag_buf_or = exec.AllocateBuffer(count * elem_bytes);
  if (!imag_buf_or.ok()) {
    LogGpuError(exec.Type(), imag_buf_or.status(), "fft1d");
    if (error) *error = imag_buf_or.status().message;
    return std::nullopt;
  }

  GpuBuffer in_buf = in_buf_or.value();
  GpuBuffer real_buf = real_buf_or.value();
  GpuBuffer imag_buf = imag_buf_or.value();
  cleanup.Add(&in_buf);
  cleanup.Add(&real_buf);
  cleanup.Add(&imag_buf);

  Status status = exec.WriteBufferFromDouble(in_buf, in.tensor.Data(), count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "fft1d");
    if (error) *error = status.message;
    return std::nullopt;
  }

  FftParams params;
  params.n = static_cast<uint64_t>(n);

  LaunchConfig cfg = Make1DLaunch(static_cast<uint64_t>(n));
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(in_buf));
  args.push_back(GpuArg::Buffer(real_buf));
  args.push_back(GpuArg::Buffer(imag_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "fft1d");
    if (error) *error = status.message;
    return std::nullopt;
  }

  auto real_or = exec.ReadBufferToDouble(real_buf, count);
  if (!real_or.ok()) {
    LogGpuError(exec.Type(), real_or.status(), "fft1d");
    if (error) *error = real_or.status().message;
    return std::nullopt;
  }
  auto imag_or = exec.ReadBufferToDouble(imag_buf, count);
  if (!imag_or.ok()) {
    LogGpuError(exec.Type(), imag_or.status(), "fft1d");
    if (error) *error = imag_or.status().message;
    return std::nullopt;
  }

  Value real_t = Value::Tensor({n}, DType::kF64, 0.0);
  Value imag_t = Value::Tensor({n}, DType::kF64, 0.0);
  for (int64_t i = 0; i < n; ++i) {
    real_t.tensor.Data()[i] = real_or.value()[static_cast<size_t>(i)];
    imag_t.tensor.Data()[i] = imag_or.value()[static_cast<size_t>(i)];
  }
  return Value::Tuple({real_t, imag_t});
}

std::optional<Value> TryGpuSolve(const Value& a, const Value& b, int line, int column,
                                 std::string* error) {
  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;
  Value A = ToDenseTensor(a, line, column);
  Value B = ToDenseTensor(b, line, column);
  if (A.type != DType::kTensor || B.type != DType::kTensor) {
    throw util::Error("solve expects tensor arguments", line, column);
  }
  if (A.tensor.shape.size() != 2) {
    throw util::Error("solve expects a 2D coefficient matrix", line, column);
  }
  int64_t n = A.tensor.shape[0];
  if (A.tensor.shape[1] != n) {
    throw util::Error("solve requires a square matrix", line, column);
  }
  int64_t rhs_cols = 0;
  if (B.tensor.shape.size() == 1) {
    if (B.tensor.shape[0] != n) {
      throw util::Error("solve rhs length must match matrix rows", line, column);
    }
    rhs_cols = 1;
  } else if (B.tensor.shape.size() == 2) {
    if (B.tensor.shape[0] != n) {
      throw util::Error("solve rhs rows must match matrix rows", line, column);
    }
    rhs_cols = B.tensor.shape[1];
  } else {
    throw util::Error("solve rhs must be 1D or 2D tensor", line, column);
  }

  auto kernel_or = exec.GetKernel({"lattice_solve", "tensor_solve"});
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "solve");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);
  size_t a_count = static_cast<size_t>(A.tensor.size);
  size_t b_count = static_cast<size_t>(B.tensor.size);
  size_t out_count = static_cast<size_t>(n * rhs_cols);

  BufferCleanup cleanup(exec);
  auto a_buf_or = exec.AllocateBuffer(a_count * elem_bytes);
  if (!a_buf_or.ok()) {
    LogGpuError(exec.Type(), a_buf_or.status(), "solve");
    if (error) *error = a_buf_or.status().message;
    return std::nullopt;
  }
  auto b_buf_or = exec.AllocateBuffer(b_count * elem_bytes);
  if (!b_buf_or.ok()) {
    LogGpuError(exec.Type(), b_buf_or.status(), "solve");
    if (error) *error = b_buf_or.status().message;
    return std::nullopt;
  }
  auto out_buf_or = exec.AllocateBuffer(out_count * elem_bytes);
  if (!out_buf_or.ok()) {
    LogGpuError(exec.Type(), out_buf_or.status(), "solve");
    if (error) *error = out_buf_or.status().message;
    return std::nullopt;
  }
  auto scratch_or = exec.AllocateBuffer(a_count * elem_bytes);
  if (!scratch_or.ok()) {
    LogGpuError(exec.Type(), scratch_or.status(), "solve");
    if (error) *error = scratch_or.status().message;
    return std::nullopt;
  }
  auto status_or = exec.AllocateBuffer(sizeof(int));
  if (!status_or.ok()) {
    LogGpuError(exec.Type(), status_or.status(), "solve");
    if (error) *error = status_or.status().message;
    return std::nullopt;
  }

  GpuBuffer a_buf = a_buf_or.value();
  GpuBuffer b_buf = b_buf_or.value();
  GpuBuffer out_buf = out_buf_or.value();
  GpuBuffer scratch_buf = scratch_or.value();
  GpuBuffer status_buf = status_or.value();
  cleanup.Add(&a_buf);
  cleanup.Add(&b_buf);
  cleanup.Add(&out_buf);
  cleanup.Add(&scratch_buf);
  cleanup.Add(&status_buf);

  Status status = exec.WriteBufferFromDouble(a_buf, A.tensor.Data(), a_count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "solve");
    if (error) *error = status.message;
    return std::nullopt;
  }
  status = exec.WriteBufferFromDouble(b_buf, B.tensor.Data(), b_count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "solve");
    if (error) *error = status.message;
    return std::nullopt;
  }
  int zero = 0;
  status = exec.WriteBuffer(status_buf, &zero, sizeof(zero));
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "solve");
    if (error) *error = status.message;
    return std::nullopt;
  }

  SolveParams params;
  params.n = static_cast<uint64_t>(n);
  params.rhs_cols = static_cast<uint64_t>(rhs_cols);

  LaunchConfig cfg = MakeSingleLaunch();
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(a_buf));
  args.push_back(GpuArg::Buffer(b_buf));
  args.push_back(GpuArg::Buffer(out_buf));
  args.push_back(GpuArg::Buffer(scratch_buf));
  args.push_back(GpuArg::Buffer(status_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "solve");
    if (error) *error = status.message;
    return std::nullopt;
  }

  int status_code = 0;
  status = exec.ReadBuffer(status_buf, &status_code, sizeof(status_code));
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "solve");
    if (error) *error = status.message;
    return std::nullopt;
  }
  if (status_code != 0) {
    throw util::Error("solve singular matrix", line, column);
  }

  auto out_data_or = exec.ReadBufferToDouble(out_buf, out_count);
  if (!out_data_or.ok()) {
    LogGpuError(exec.Type(), out_data_or.status(), "solve");
    if (error) *error = out_data_or.status().message;
    return std::nullopt;
  }

  DType elem = PromoteType(A.tensor.elem_type, B.tensor.elem_type, line, column);
  auto out_dense = BuildDenseTensor({n, rhs_cols}, elem, out_data_or.value());
  if (!out_dense.has_value()) return std::nullopt;
  if (rhs_cols == 1) {
    out_dense->tensor.shape = {n};
    out_dense->tensor.strides = {1};
    out_dense->tensor.size = n;
  }
  return out_dense.value();
}

std::optional<Value> TryGpuLu(const Value& a, int line, int column, std::string* error) {
  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;
  Value A = ToDenseTensor(a, line, column);
  if (A.type != DType::kTensor || A.tensor.shape.size() != 2 ||
      A.tensor.shape[0] != A.tensor.shape[1]) {
    throw util::Error("lu expects a square 2D tensor", line, column);
  }
  int64_t n = A.tensor.shape[0];

  auto kernel_or = exec.GetKernel({"lattice_lu", "tensor_lu"});
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "lu");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);
  size_t a_count = static_cast<size_t>(A.tensor.size);
  size_t out_count = static_cast<size_t>(n * n);

  BufferCleanup cleanup(exec);
  auto a_buf_or = exec.AllocateBuffer(a_count * elem_bytes);
  if (!a_buf_or.ok()) {
    LogGpuError(exec.Type(), a_buf_or.status(), "lu");
    if (error) *error = a_buf_or.status().message;
    return std::nullopt;
  }
  auto l_buf_or = exec.AllocateBuffer(out_count * elem_bytes);
  if (!l_buf_or.ok()) {
    LogGpuError(exec.Type(), l_buf_or.status(), "lu");
    if (error) *error = l_buf_or.status().message;
    return std::nullopt;
  }
  auto u_buf_or = exec.AllocateBuffer(out_count * elem_bytes);
  if (!u_buf_or.ok()) {
    LogGpuError(exec.Type(), u_buf_or.status(), "lu");
    if (error) *error = u_buf_or.status().message;
    return std::nullopt;
  }
  auto status_or = exec.AllocateBuffer(sizeof(int));
  if (!status_or.ok()) {
    LogGpuError(exec.Type(), status_or.status(), "lu");
    if (error) *error = status_or.status().message;
    return std::nullopt;
  }

  GpuBuffer a_buf = a_buf_or.value();
  GpuBuffer l_buf = l_buf_or.value();
  GpuBuffer u_buf = u_buf_or.value();
  GpuBuffer status_buf = status_or.value();
  cleanup.Add(&a_buf);
  cleanup.Add(&l_buf);
  cleanup.Add(&u_buf);
  cleanup.Add(&status_buf);

  Status status = exec.WriteBufferFromDouble(a_buf, A.tensor.Data(), a_count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "lu");
    if (error) *error = status.message;
    return std::nullopt;
  }
  int zero = 0;
  status = exec.WriteBuffer(status_buf, &zero, sizeof(zero));
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "lu");
    if (error) *error = status.message;
    return std::nullopt;
  }

  LuParams params;
  params.n = static_cast<uint64_t>(n);

  LaunchConfig cfg = MakeSingleLaunch();
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(a_buf));
  args.push_back(GpuArg::Buffer(l_buf));
  args.push_back(GpuArg::Buffer(u_buf));
  args.push_back(GpuArg::Buffer(status_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "lu");
    if (error) *error = status.message;
    return std::nullopt;
  }

  int status_code = 0;
  status = exec.ReadBuffer(status_buf, &status_code, sizeof(status_code));
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "lu");
    if (error) *error = status.message;
    return std::nullopt;
  }
  if (status_code != 0) {
    throw util::Error("lu singular matrix", line, column);
  }

  auto l_or = exec.ReadBufferToDouble(l_buf, out_count);
  if (!l_or.ok()) {
    LogGpuError(exec.Type(), l_or.status(), "lu");
    if (error) *error = l_or.status().message;
    return std::nullopt;
  }
  auto u_or = exec.ReadBufferToDouble(u_buf, out_count);
  if (!u_or.ok()) {
    LogGpuError(exec.Type(), u_or.status(), "lu");
    if (error) *error = u_or.status().message;
    return std::nullopt;
  }

  auto l_val = BuildDenseTensor({n, n}, A.tensor.elem_type, l_or.value());
  auto u_val = BuildDenseTensor({n, n}, A.tensor.elem_type, u_or.value());
  if (!l_val.has_value() || !u_val.has_value()) return std::nullopt;
  return Value::Tuple({l_val.value(), u_val.value()});
}

std::optional<Value> TryGpuQr(const Value& a, int line, int column, std::string* error) {
  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;
  Value A = ToDenseTensor(a, line, column);
  if (A.type != DType::kTensor || A.tensor.shape.size() != 2) {
    throw util::Error("qr expects a 2D tensor", line, column);
  }
  int64_t m = A.tensor.shape[0];
  int64_t n = A.tensor.shape[1];

  auto kernel_or = exec.GetKernel({"lattice_qr", "tensor_qr"});
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "qr");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);
  size_t a_count = static_cast<size_t>(A.tensor.size);
  size_t q_count = static_cast<size_t>(m * n);
  size_t r_count = static_cast<size_t>(n * n);

  BufferCleanup cleanup(exec);
  auto a_buf_or = exec.AllocateBuffer(a_count * elem_bytes);
  if (!a_buf_or.ok()) {
    LogGpuError(exec.Type(), a_buf_or.status(), "qr");
    if (error) *error = a_buf_or.status().message;
    return std::nullopt;
  }
  auto q_buf_or = exec.AllocateBuffer(q_count * elem_bytes);
  if (!q_buf_or.ok()) {
    LogGpuError(exec.Type(), q_buf_or.status(), "qr");
    if (error) *error = q_buf_or.status().message;
    return std::nullopt;
  }
  auto r_buf_or = exec.AllocateBuffer(r_count * elem_bytes);
  if (!r_buf_or.ok()) {
    LogGpuError(exec.Type(), r_buf_or.status(), "qr");
    if (error) *error = r_buf_or.status().message;
    return std::nullopt;
  }
  auto status_or = exec.AllocateBuffer(sizeof(int));
  if (!status_or.ok()) {
    LogGpuError(exec.Type(), status_or.status(), "qr");
    if (error) *error = status_or.status().message;
    return std::nullopt;
  }

  GpuBuffer a_buf = a_buf_or.value();
  GpuBuffer q_buf = q_buf_or.value();
  GpuBuffer r_buf = r_buf_or.value();
  GpuBuffer status_buf = status_or.value();
  cleanup.Add(&a_buf);
  cleanup.Add(&q_buf);
  cleanup.Add(&r_buf);
  cleanup.Add(&status_buf);

  Status status = exec.WriteBufferFromDouble(a_buf, A.tensor.Data(), a_count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "qr");
    if (error) *error = status.message;
    return std::nullopt;
  }
  int zero = 0;
  status = exec.WriteBuffer(status_buf, &zero, sizeof(zero));
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "qr");
    if (error) *error = status.message;
    return std::nullopt;
  }

  QrParams params;
  params.m = static_cast<uint64_t>(m);
  params.n = static_cast<uint64_t>(n);

  LaunchConfig cfg = MakeSingleLaunch();
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(a_buf));
  args.push_back(GpuArg::Buffer(q_buf));
  args.push_back(GpuArg::Buffer(r_buf));
  args.push_back(GpuArg::Buffer(status_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "qr");
    if (error) *error = status.message;
    return std::nullopt;
  }

  int status_code = 0;
  status = exec.ReadBuffer(status_buf, &status_code, sizeof(status_code));
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "qr");
    if (error) *error = status.message;
    return std::nullopt;
  }
  if (status_code != 0) {
    throw util::Error("qr rank deficiency", line, column);
  }

  auto q_or = exec.ReadBufferToDouble(q_buf, q_count);
  if (!q_or.ok()) {
    LogGpuError(exec.Type(), q_or.status(), "qr");
    if (error) *error = q_or.status().message;
    return std::nullopt;
  }
  auto r_or = exec.ReadBufferToDouble(r_buf, r_count);
  if (!r_or.ok()) {
    LogGpuError(exec.Type(), r_or.status(), "qr");
    if (error) *error = r_or.status().message;
    return std::nullopt;
  }

  auto q_val = BuildDenseTensor({m, n}, A.tensor.elem_type, q_or.value());
  auto r_val = BuildDenseTensor({n, n}, A.tensor.elem_type, r_or.value());
  if (!q_val.has_value() || !r_val.has_value()) return std::nullopt;
  return Value::Tuple({q_val.value(), r_val.value()});
}

std::optional<Value> TryGpuSvd(const Value& a, int line, int column, std::string* error) {
  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;
  Value A = ToDenseTensor(a, line, column);
  if (A.type != DType::kTensor || A.tensor.shape.size() != 2) {
    throw util::Error("svd expects a 2D tensor", line, column);
  }
  int64_t m = A.tensor.shape[0];
  int64_t n = A.tensor.shape[1];
  if (m != 2 || n != 2) {
    throw util::Error("svd currently supports 2x2 matrices only", line, column);
  }

  auto kernel_or = exec.GetKernel({"lattice_svd", "tensor_svd"});
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "svd");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);
  size_t a_count = static_cast<size_t>(A.tensor.size);
  size_t out_count = static_cast<size_t>(m * n);

  BufferCleanup cleanup(exec);
  auto a_buf_or = exec.AllocateBuffer(a_count * elem_bytes);
  if (!a_buf_or.ok()) {
    LogGpuError(exec.Type(), a_buf_or.status(), "svd");
    if (error) *error = a_buf_or.status().message;
    return std::nullopt;
  }
  auto u_buf_or = exec.AllocateBuffer(out_count * elem_bytes);
  if (!u_buf_or.ok()) {
    LogGpuError(exec.Type(), u_buf_or.status(), "svd");
    if (error) *error = u_buf_or.status().message;
    return std::nullopt;
  }
  auto s_buf_or = exec.AllocateBuffer(out_count * elem_bytes);
  if (!s_buf_or.ok()) {
    LogGpuError(exec.Type(), s_buf_or.status(), "svd");
    if (error) *error = s_buf_or.status().message;
    return std::nullopt;
  }
  auto v_buf_or = exec.AllocateBuffer(out_count * elem_bytes);
  if (!v_buf_or.ok()) {
    LogGpuError(exec.Type(), v_buf_or.status(), "svd");
    if (error) *error = v_buf_or.status().message;
    return std::nullopt;
  }

  GpuBuffer a_buf = a_buf_or.value();
  GpuBuffer u_buf = u_buf_or.value();
  GpuBuffer s_buf = s_buf_or.value();
  GpuBuffer v_buf = v_buf_or.value();
  cleanup.Add(&a_buf);
  cleanup.Add(&u_buf);
  cleanup.Add(&s_buf);
  cleanup.Add(&v_buf);

  Status status = exec.WriteBufferFromDouble(a_buf, A.tensor.Data(), a_count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "svd");
    if (error) *error = status.message;
    return std::nullopt;
  }

  SvdParams params;
  params.m = static_cast<uint64_t>(m);
  params.n = static_cast<uint64_t>(n);

  LaunchConfig cfg = MakeSingleLaunch();
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(a_buf));
  args.push_back(GpuArg::Buffer(u_buf));
  args.push_back(GpuArg::Buffer(s_buf));
  args.push_back(GpuArg::Buffer(v_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "svd");
    if (error) *error = status.message;
    return std::nullopt;
  }

  auto u_or = exec.ReadBufferToDouble(u_buf, out_count);
  if (!u_or.ok()) {
    LogGpuError(exec.Type(), u_or.status(), "svd");
    if (error) *error = u_or.status().message;
    return std::nullopt;
  }
  auto s_or = exec.ReadBufferToDouble(s_buf, out_count);
  if (!s_or.ok()) {
    LogGpuError(exec.Type(), s_or.status(), "svd");
    if (error) *error = s_or.status().message;
    return std::nullopt;
  }
  auto v_or = exec.ReadBufferToDouble(v_buf, out_count);
  if (!v_or.ok()) {
    LogGpuError(exec.Type(), v_or.status(), "svd");
    if (error) *error = v_or.status().message;
    return std::nullopt;
  }

  auto u_val = BuildDenseTensor({m, n}, A.tensor.elem_type, u_or.value());
  auto s_val = BuildDenseTensor({m, n}, A.tensor.elem_type, s_or.value());
  auto v_val = BuildDenseTensor({m, n}, A.tensor.elem_type, v_or.value());
  if (!u_val.has_value() || !s_val.has_value() || !v_val.has_value()) return std::nullopt;
  return Value::Tuple({u_val.value(), s_val.value(), v_val.value()});
}

std::optional<Value> TryGpuQuantile(const Value& data, double q, int line, int column,
                                    std::string* error) {
  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;
  if (q < 0.0 || q > 1.0) {
    throw util::Error("quantile q must be in [0,1]", line, column);
  }
  if (data.type != DType::kTensor) return std::nullopt;
  Value dense = ToDenseTensor(data, line, column);
  if (dense.tensor.shape.size() != 1) {
    throw util::Error("Expected a 1D tensor", line, column);
  }
  int64_t count = dense.tensor.shape[0];
  if (count <= 0) {
    throw util::Error("quantile of empty data", line, column);
  }

  auto kernel_or = exec.GetKernel({"lattice_quantile", "tensor_quantile"});
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "quantile");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);
  size_t in_count = static_cast<size_t>(count);

  BufferCleanup cleanup(exec);
  auto in_buf_or = exec.AllocateBuffer(in_count * elem_bytes);
  if (!in_buf_or.ok()) {
    LogGpuError(exec.Type(), in_buf_or.status(), "quantile");
    if (error) *error = in_buf_or.status().message;
    return std::nullopt;
  }
  auto scratch_or = exec.AllocateBuffer(in_count * elem_bytes);
  if (!scratch_or.ok()) {
    LogGpuError(exec.Type(), scratch_or.status(), "quantile");
    if (error) *error = scratch_or.status().message;
    return std::nullopt;
  }
  auto out_buf_or = exec.AllocateBuffer(elem_bytes);
  if (!out_buf_or.ok()) {
    LogGpuError(exec.Type(), out_buf_or.status(), "quantile");
    if (error) *error = out_buf_or.status().message;
    return std::nullopt;
  }

  GpuBuffer in_buf = in_buf_or.value();
  GpuBuffer scratch_buf = scratch_or.value();
  GpuBuffer out_buf = out_buf_or.value();
  cleanup.Add(&in_buf);
  cleanup.Add(&scratch_buf);
  cleanup.Add(&out_buf);

  Status status = exec.WriteBufferFromDouble(in_buf, dense.tensor.Data(), in_count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "quantile");
    if (error) *error = status.message;
    return std::nullopt;
  }

  QuantileParams params;
  params.count = static_cast<uint64_t>(in_count);
  params.q = static_cast<float>(q);
  params.pad = 0;

  LaunchConfig cfg = MakeSingleLaunch();
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(in_buf));
  args.push_back(GpuArg::Buffer(scratch_buf));
  args.push_back(GpuArg::Buffer(out_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "quantile");
    if (error) *error = status.message;
    return std::nullopt;
  }

  auto out_data_or = exec.ReadBufferToDouble(out_buf, 1);
  if (!out_data_or.ok()) {
    LogGpuError(exec.Type(), out_data_or.status(), "quantile");
    if (error) *error = out_data_or.status().message;
    return std::nullopt;
  }

  return Value::F64(out_data_or.value()[0]);
}

std::optional<Value> TryGpuCorrelation(const Value& x, const Value& y, int line, int column,
                                       std::string* error) {
  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;
  if (x.type != DType::kTensor || y.type != DType::kTensor) {
    throw util::Error("correlation requires equal non-empty vectors", line, column);
  }
  Value dx = ToDenseTensor(x, line, column);
  Value dy = ToDenseTensor(y, line, column);
  if (dx.tensor.shape.size() != 1 || dy.tensor.shape.size() != 1 ||
      dx.tensor.shape[0] != dy.tensor.shape[0] || dx.tensor.shape[0] == 0) {
    throw util::Error("correlation requires equal non-empty vectors", line, column);
  }
  size_t count = static_cast<size_t>(dx.tensor.shape[0]);

  auto kernel_or = exec.GetKernel({"lattice_correlation", "tensor_correlation"});
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "correlation");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);

  BufferCleanup cleanup(exec);
  auto x_buf_or = exec.AllocateBuffer(count * elem_bytes);
  if (!x_buf_or.ok()) {
    LogGpuError(exec.Type(), x_buf_or.status(), "correlation");
    if (error) *error = x_buf_or.status().message;
    return std::nullopt;
  }
  auto y_buf_or = exec.AllocateBuffer(count * elem_bytes);
  if (!y_buf_or.ok()) {
    LogGpuError(exec.Type(), y_buf_or.status(), "correlation");
    if (error) *error = y_buf_or.status().message;
    return std::nullopt;
  }
  auto out_buf_or = exec.AllocateBuffer(elem_bytes);
  if (!out_buf_or.ok()) {
    LogGpuError(exec.Type(), out_buf_or.status(), "correlation");
    if (error) *error = out_buf_or.status().message;
    return std::nullopt;
  }
  auto status_or = exec.AllocateBuffer(sizeof(int));
  if (!status_or.ok()) {
    LogGpuError(exec.Type(), status_or.status(), "correlation");
    if (error) *error = status_or.status().message;
    return std::nullopt;
  }

  GpuBuffer x_buf = x_buf_or.value();
  GpuBuffer y_buf = y_buf_or.value();
  GpuBuffer out_buf = out_buf_or.value();
  GpuBuffer status_buf = status_or.value();
  cleanup.Add(&x_buf);
  cleanup.Add(&y_buf);
  cleanup.Add(&out_buf);
  cleanup.Add(&status_buf);

  Status status = exec.WriteBufferFromDouble(x_buf, dx.tensor.Data(), count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "correlation");
    if (error) *error = status.message;
    return std::nullopt;
  }
  status = exec.WriteBufferFromDouble(y_buf, dy.tensor.Data(), count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "correlation");
    if (error) *error = status.message;
    return std::nullopt;
  }
  int zero = 0;
  status = exec.WriteBuffer(status_buf, &zero, sizeof(zero));
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "correlation");
    if (error) *error = status.message;
    return std::nullopt;
  }

  CorrelationParams params;
  params.count = static_cast<uint64_t>(count);

  LaunchConfig cfg = MakeSingleLaunch();
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(x_buf));
  args.push_back(GpuArg::Buffer(y_buf));
  args.push_back(GpuArg::Buffer(out_buf));
  args.push_back(GpuArg::Buffer(status_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "correlation");
    if (error) *error = status.message;
    return std::nullopt;
  }

  int status_code = 0;
  status = exec.ReadBuffer(status_buf, &status_code, sizeof(status_code));
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "correlation");
    if (error) *error = status.message;
    return std::nullopt;
  }
  if (status_code != 0) {
    throw util::Error("correlation undefined (zero variance)", line, column);
  }

  auto out_data_or = exec.ReadBufferToDouble(out_buf, 1);
  if (!out_data_or.ok()) {
    LogGpuError(exec.Type(), out_data_or.status(), "correlation");
    if (error) *error = out_data_or.status().message;
    return std::nullopt;
  }
  return Value::F64(out_data_or.value()[0]);
}

std::optional<Value> TryGpuRegression(const Value& x, const Value& y, int line, int column,
                                      std::string* error) {
  auto& exec = GpuExecutor::Instance();
  if (!exec.Enabled()) return std::nullopt;
  if (x.type != DType::kTensor || y.type != DType::kTensor) {
    throw util::Error("regression requires equal non-empty vectors", line, column);
  }
  Value dx = ToDenseTensor(x, line, column);
  Value dy = ToDenseTensor(y, line, column);
  if (dx.tensor.shape.size() != 1 || dy.tensor.shape.size() != 1 ||
      dx.tensor.shape[0] != dy.tensor.shape[0] || dx.tensor.shape[0] == 0) {
    throw util::Error("regression requires equal non-empty vectors", line, column);
  }
  size_t count = static_cast<size_t>(dx.tensor.shape[0]);

  auto kernel_or = exec.GetKernel({"lattice_regression", "tensor_regression"});
  if (!kernel_or.ok()) {
    LogGpuError(exec.Type(), kernel_or.status(), "regression");
    if (error) *error = kernel_or.status().message;
    return std::nullopt;
  }

  size_t elem_bytes = exec.UseFp64() ? sizeof(double) : sizeof(float);

  BufferCleanup cleanup(exec);
  auto x_buf_or = exec.AllocateBuffer(count * elem_bytes);
  if (!x_buf_or.ok()) {
    LogGpuError(exec.Type(), x_buf_or.status(), "regression");
    if (error) *error = x_buf_or.status().message;
    return std::nullopt;
  }
  auto y_buf_or = exec.AllocateBuffer(count * elem_bytes);
  if (!y_buf_or.ok()) {
    LogGpuError(exec.Type(), y_buf_or.status(), "regression");
    if (error) *error = y_buf_or.status().message;
    return std::nullopt;
  }
  auto out_buf_or = exec.AllocateBuffer(2 * elem_bytes);
  if (!out_buf_or.ok()) {
    LogGpuError(exec.Type(), out_buf_or.status(), "regression");
    if (error) *error = out_buf_or.status().message;
    return std::nullopt;
  }
  auto status_or = exec.AllocateBuffer(sizeof(int));
  if (!status_or.ok()) {
    LogGpuError(exec.Type(), status_or.status(), "regression");
    if (error) *error = status_or.status().message;
    return std::nullopt;
  }

  GpuBuffer x_buf = x_buf_or.value();
  GpuBuffer y_buf = y_buf_or.value();
  GpuBuffer out_buf = out_buf_or.value();
  GpuBuffer status_buf = status_or.value();
  cleanup.Add(&x_buf);
  cleanup.Add(&y_buf);
  cleanup.Add(&out_buf);
  cleanup.Add(&status_buf);

  Status status = exec.WriteBufferFromDouble(x_buf, dx.tensor.Data(), count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "regression");
    if (error) *error = status.message;
    return std::nullopt;
  }
  status = exec.WriteBufferFromDouble(y_buf, dy.tensor.Data(), count);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "regression");
    if (error) *error = status.message;
    return std::nullopt;
  }
  int zero = 0;
  status = exec.WriteBuffer(status_buf, &zero, sizeof(zero));
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "regression");
    if (error) *error = status.message;
    return std::nullopt;
  }

  RegressionParams params;
  params.count = static_cast<uint64_t>(count);

  LaunchConfig cfg = MakeSingleLaunch();
  std::vector<GpuArg> args;
  args.push_back(GpuArg::Buffer(x_buf));
  args.push_back(GpuArg::Buffer(y_buf));
  args.push_back(GpuArg::Buffer(out_buf));
  args.push_back(GpuArg::Buffer(status_buf));
  args.push_back(GpuArg::Value(&params, sizeof(params)));

  status = exec.LaunchKernel(kernel_or.value(), cfg, args);
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "regression");
    if (error) *error = status.message;
    return std::nullopt;
  }

  int status_code = 0;
  status = exec.ReadBuffer(status_buf, &status_code, sizeof(status_code));
  if (!status.ok()) {
    LogGpuError(exec.Type(), status, "regression");
    if (error) *error = status.message;
    return std::nullopt;
  }
  if (status_code != 0) {
    throw util::Error("regression undefined (zero variance)", line, column);
  }

  auto out_data_or = exec.ReadBufferToDouble(out_buf, 2);
  if (!out_data_or.ok()) {
    LogGpuError(exec.Type(), out_data_or.status(), "regression");
    if (error) *error = out_data_or.status().message;
    return std::nullopt;
  }

  Value slope = Value::F64(out_data_or.value()[0]);
  Value intercept = Value::F64(out_data_or.value()[1]);
  return Value::Tuple({slope, intercept});
}

}  // namespace lattice::runtime
