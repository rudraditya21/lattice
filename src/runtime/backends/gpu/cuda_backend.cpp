#include "runtime/backends/cuda_backend.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/backends/backend_error.h"
#include "runtime/backends/backend_log.h"
#include "runtime/backends/cache_store.h"
#include "runtime/backends/cuda_abi.h"
#include "runtime/backends/device_quirks.h"
#include "runtime/backends/device_selector.h"
#include "runtime/backends/memory_pool.h"

namespace lattice::runtime {

namespace {

constexpr int kAttrMaxThreadsPerBlock = 1;
constexpr int kAttrSharedMemPerBlock = 8;
constexpr int kAttrClockRate = 13;  // kHz
constexpr int kAttrMultiprocessorCount = 16;

std::string FormatCudaVersion(int version) {
  if (version <= 0) return "";
  int major = version / 1000;
  int minor = (version % 1000) / 10;
  std::ostringstream ss;
  ss << major << "." << minor;
  return ss.str();
}

std::string FormatRuntimeVersion(int major, int minor) {
  if (major <= 0 && minor <= 0) return "";
  std::ostringstream ss;
  ss << major << "." << minor;
  return ss.str();
}

uint64_t Fnv1a64(const std::string& data) {
  uint64_t hash = 14695981039346656037ull;
  for (unsigned char c : data) {
    hash ^= static_cast<uint64_t>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string Hex64(uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

Status CudaStatus(StatusCode code, BackendErrorKind kind, const std::string& message) {
  return MakeBackendError(code, BackendType::kCUDA, kind, message);
}

int CapabilityToInt(CapabilityStatus status) {
  switch (status) {
    case CapabilityStatus::kYes:
      return 1;
    case CapabilityStatus::kNo:
      return 0;
    default:
      return -1;
  }
}

bool ReadFile(const std::filesystem::path& path, std::string* out, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error) {
      *error = "Failed to open file: " + path.string();
    }
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  *out = ss.str();
  return true;
}

bool IsBinarySource(const std::filesystem::path& path) {
  auto ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext == ".ptx" || ext == ".cubin" || ext == ".fatbin";
}

std::vector<std::string> SplitOptions(const std::string& options) {
  std::vector<std::string> out;
  std::string current;
  bool in_quote = false;
  for (char c : options) {
    if (c == '"') {
      in_quote = !in_quote;
      continue;
    }
    if (!in_quote && std::isspace(static_cast<unsigned char>(c))) {
      if (!current.empty()) {
        out.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) out.push_back(current);
  return out;
}

DeviceMetadata BuildDeviceMetadata(const CudaDeviceDesc& desc, const DeviceCapabilities& caps) {
  DeviceMetadata meta;
  meta.backend = "cuda";
  meta.index = desc.index;
  meta.name = desc.name;
  meta.vendor = desc.vendor;
  meta.driver_version = desc.driver_version;
  meta.runtime_version = desc.runtime_version;
  meta.device_version = std::to_string(desc.major) + "." + std::to_string(desc.minor);
  meta.total_mem = desc.total_mem;
  meta.multiprocessor_count = desc.multiprocessor_count;
  meta.clock_khz = desc.clock_khz;
  meta.is_gpu = caps.is_gpu;
  meta.is_cpu = caps.is_cpu;
  meta.is_accel = false;
  meta.fp16 = CapabilityToInt(caps.fp16);
  meta.fp64 = CapabilityToInt(caps.fp64);
  return meta;
}

CacheStore& CudaCacheStore() {
  static CacheStore store("cuda");
  return store;
}

constexpr unsigned int kCuMemHostAllocPortable = 0x01;

MemoryPoolConfig CudaDevicePoolConfig() {
  static MemoryPoolConfig config = [] {
    MemoryPoolConfig base = DefaultDevicePoolConfig();
    base = LoadMemoryPoolConfig("LATTICE_DEVICE_POOL", base);
    base = LoadMemoryPoolConfig("LATTICE_CUDA_DEVICE_POOL", base);
    return base;
  }();
  return config;
}

MemoryPoolConfig CudaPinnedPoolConfig() {
  static MemoryPoolConfig config = [] {
    MemoryPoolConfig base = DefaultPinnedPoolConfig();
    base = LoadMemoryPoolConfig("LATTICE_PINNED_POOL", base);
    base = LoadMemoryPoolConfig("LATTICE_CUDA_PINNED_POOL", base);
    return base;
  }();
  return config;
}

class CudaEvent final : public Event {
 public:
  void Record() override { ready_ = true; }
  void Wait() override {}
  bool Ready() const override { return ready_; }

 private:
  bool ready_ = false;
};

class CudaStream final : public Stream {
 public:
  CudaStream(const gpu::CudaLoader* loader, gpu::CUstream stream)
      : loader_(loader), stream_(stream) {}

  void Submit(std::function<void()> fn) override { fn(); }

  void Synchronize() override {
    if (loader_ && loader_->cuStreamSynchronize) {
      loader_->cuStreamSynchronize(stream_);
    }
  }

  void AddDependency(const std::shared_ptr<Event>&) override {}
  void SetPriority(int) override {}

 private:
  const gpu::CudaLoader* loader_ = nullptr;
  gpu::CUstream stream_ = nullptr;
};

}  // namespace

struct CudaBackend::DeviceContext {
  gpu::CUdevice device = 0;
  gpu::CUcontext context = nullptr;
  gpu::CUstream stream = nullptr;
  CudaDeviceDesc desc;
  DeviceCapabilities caps;
  std::string fingerprint;
  std::unordered_map<std::string, gpu::CUmodule> module_cache;
  std::unique_ptr<MemoryPool> device_pool;
};

CudaBackend::CudaBackend() = default;

CudaBackend::~CudaBackend() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& dev : devices_) {
    if (loader_.cuCtxSetCurrent && dev.context) {
      loader_.cuCtxSetCurrent(dev.context);
    }
    if (dev.device_pool) {
      if (dev.device_pool->Outstanding() > 0) {
        LogBackend({LogLevel::kWarn, BackendType::kCUDA, BackendErrorKind::kMemory,
                    "device pool has outstanding allocations", "device_pool_leak", dev.desc.index,
                    dev.desc.name});
      }
      dev.device_pool->Trim();
    }
    if (loader_.cuStreamDestroy && dev.stream) {
      loader_.cuStreamDestroy(dev.stream);
      dev.stream = nullptr;
    }
    if (loader_.cuCtxDestroy && dev.context) {
      loader_.cuCtxDestroy(dev.context);
      dev.context = nullptr;
    }
  }
  if (pinned_pool_) {
    if (pinned_pool_->Outstanding() > 0) {
      LogBackend({LogLevel::kWarn, BackendType::kCUDA, BackendErrorKind::kMemory,
                  "pinned pool has outstanding allocations", "pinned_pool_leak", -1, "cuda"});
    }
    pinned_pool_->Trim();
  }
  loader_.Unload();
}

BackendType CudaBackend::Type() const {
  return BackendType::kCUDA;
}

std::string CudaBackend::Name() const {
  return "CUDA";
}

BackendCapabilities CudaBackend::Capabilities() const {
  BackendCapabilities caps;
  caps.supports_dense = true;
  caps.supports_sparse = false;
  caps.supports_ragged = false;
  caps.supports_fft = false;
  caps.supports_blas = false;
  caps.supports_conv = false;
  caps.supports_rng = true;
  caps.supports_events = true;
  caps.supported_dtypes = {DType::kF32, DType::kF64, DType::kI32, DType::kU32};
  return caps;
}

StatusOr<std::shared_ptr<Stream>> CudaBackend::CreateStream() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) {
    return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                      "No CUDA devices available");
  }
  return std::make_shared<CudaStream>(&loader_, devices_[0].stream);
}

StatusOr<std::shared_ptr<Event>> CudaBackend::CreateEvent() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  return std::make_shared<CudaEvent>();
}

StatusOr<Allocation> CudaBackend::Allocate(size_t bytes, size_t alignment) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) {
    return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                      "No CUDA devices available");
  }
  if (bytes == 0) {
    Allocation empty;
    empty.kind = AllocationKind::kDevice;
    return empty;
  }
  auto* pool = DevicePool(0);
  if (!pool) {
    return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                      "Device pool unavailable");
  }
  auto block_or = pool->Acquire(bytes, alignment);
  if (!block_or.ok()) return block_or.status();
  auto block = block_or.value();
  Allocation alloc;
  alloc.ptr = nullptr;
  alloc.device_handle = reinterpret_cast<void*>(block.handle);
  alloc.bytes = bytes;
  alloc.alignment = alignment;
  alloc.from_pool = block.from_pool;
  alloc.kind = AllocationKind::kDevice;
  return alloc;
}

Status CudaBackend::Deallocate(const Allocation& alloc) const {
  if (!alloc.device_handle) return Status::OK();
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  auto* pool = DevicePool(0);
  if (!pool) {
    return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                      "Device pool unavailable");
  }
  return pool->Release(reinterpret_cast<uintptr_t>(alloc.device_handle));
}

StatusOr<Allocation> CudaBackend::AllocatePinned(size_t bytes, size_t alignment) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (bytes == 0) {
    Allocation empty;
    empty.kind = AllocationKind::kPinnedHost;
    return empty;
  }
  auto* pool = PinnedPool();
  if (!pool) {
    return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                      "Pinned pool unavailable");
  }
  auto block_or = pool->Acquire(bytes, alignment);
  if (!block_or.ok()) return block_or.status();
  auto block = block_or.value();
  Allocation alloc;
  alloc.ptr = block.host_ptr;
  alloc.device_handle = reinterpret_cast<void*>(block.handle);
  alloc.bytes = bytes;
  alloc.alignment = alignment;
  alloc.from_pool = block.from_pool;
  alloc.kind = AllocationKind::kPinnedHost;
  return alloc;
}

Status CudaBackend::DeallocatePinned(const Allocation& alloc) const {
  if (!alloc.ptr) return Status::OK();
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  auto* pool = PinnedPool();
  if (!pool) {
    return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                      "Pinned pool unavailable");
  }
  return pool->Release(reinterpret_cast<uintptr_t>(alloc.ptr));
}

int CudaBackend::NumThreads() const {
  return 1;
}

size_t CudaBackend::OutstandingAllocs() const {
  size_t total = 0;
  for (int i = 0; i < static_cast<int>(devices_.size()); ++i) {
    auto& dev = devices_[i];
    if (dev.device_pool) total += dev.device_pool->Outstanding();
  }
  if (pinned_pool_) total += pinned_pool_->Outstanding();
  return total;
}

BackendMemoryStats CudaBackend::MemoryStats() const {
  BackendMemoryStats stats;
  Status status = EnsureInitialized();
  if (!status.ok()) return stats;
  for (int i = 0; i < static_cast<int>(devices_.size()); ++i) {
    auto& dev = devices_[i];
    if (dev.device_pool) {
      AccumulateMemoryPoolStats(&stats.device, dev.device_pool->Stats());
    }
  }
  if (pinned_pool_) {
    AccumulateMemoryPoolStats(&stats.pinned, pinned_pool_->Stats());
  }
  return stats;
}

void CudaBackend::SetDefaultPriority(int priority) {
  default_priority_ = priority;
}

void CudaBackend::SetDeterministic(bool deterministic) {
  deterministic_ = deterministic;
}

int CudaBackend::DeviceCount() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return 0;
  return static_cast<int>(devices_.size());
}

std::vector<CudaDeviceDesc> CudaBackend::DeviceInfo() const {
  Status status = EnsureInitialized();
  std::vector<CudaDeviceDesc> out;
  if (!status.ok()) return out;
  for (const auto& dev : devices_) {
    out.push_back(dev.desc);
  }
  return out;
}

std::vector<DeviceCapabilities> CudaBackend::DeviceCaps() const {
  Status status = EnsureInitialized();
  std::vector<DeviceCapabilities> out;
  if (!status.ok()) return out;
  for (const auto& dev : devices_) {
    out.push_back(dev.caps);
  }
  return out;
}

StatusOr<CudaBuffer> CudaBackend::CreateBuffer(int device_index, size_t bytes) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return CudaStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                      "Invalid CUDA device index");
  }
  if (bytes == 0) {
    return CudaBuffer{0, 0, device_index};
  }
  auto* pool = DevicePool(device_index);
  if (!pool) {
    return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                      "Device pool unavailable");
  }
  auto block_or = pool->Acquire(bytes, 64);
  if (!block_or.ok()) return block_or.status();
  auto block = block_or.value();
  return CudaBuffer{static_cast<gpu::CUdeviceptr>(block.handle), bytes, device_index};
}

Status CudaBackend::ReleaseBuffer(CudaBuffer* buffer) const {
  if (!buffer || !buffer->ptr) return Status::OK();
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  auto* pool = DevicePool(buffer->device_index);
  if (!pool) {
    return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                      "Device pool unavailable");
  }
  Status release = pool->Release(static_cast<uintptr_t>(buffer->ptr));
  if (!release.ok()) return release;
  buffer->ptr = 0;
  buffer->bytes = 0;
  buffer->device_index = -1;
  return Status::OK();
}

Status CudaBackend::WriteBuffer(int device_index, const CudaBuffer& buffer, const void* data,
                                size_t bytes, size_t offset) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return CudaStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                      "Invalid CUDA device index");
  }
  if (bytes + offset > buffer.bytes) {
    return CudaStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                      "Write exceeds buffer size");
  }
  auto& dev = devices_[device_index];
  if (loader_.cuCtxSetCurrent) {
    loader_.cuCtxSetCurrent(dev.context);
  }
  gpu::CUresult err = loader_.cuMemcpyHtoD(buffer.ptr + offset, data, bytes);
  if (err != gpu::CUDA_SUCCESS) {
    return CudaStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                      "cuMemcpyHtoD failed: " + gpu::CudaErrorString(err, &loader_));
  }
  return Status::OK();
}

Status CudaBackend::ReadBuffer(int device_index, const CudaBuffer& buffer, void* data, size_t bytes,
                               size_t offset) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return CudaStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                      "Invalid CUDA device index");
  }
  if (bytes + offset > buffer.bytes) {
    return CudaStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                      "Read exceeds buffer size");
  }
  auto& dev = devices_[device_index];
  if (loader_.cuCtxSetCurrent) {
    loader_.cuCtxSetCurrent(dev.context);
  }
  gpu::CUresult err = loader_.cuMemcpyDtoH(data, buffer.ptr + offset, bytes);
  if (err != gpu::CUDA_SUCCESS) {
    return CudaStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                      "cuMemcpyDtoH failed: " + gpu::CudaErrorString(err, &loader_));
  }
  return Status::OK();
}

StatusOr<CudaKernel> CudaBackend::BuildKernelFromFile(
    const std::string& path, const std::string& kernel_name,
    const std::string& extra_build_options) const {
  auto kernels_or = BuildKernelsFromFile(path, kernel_name, extra_build_options);
  if (!kernels_or.ok()) return kernels_or.status();
  auto kernels = kernels_or.value();
  if (kernels.empty()) {
    return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kBuild, "No CUDA kernels built");
  }
  return kernels.front();
}

StatusOr<std::vector<CudaKernel>> CudaBackend::BuildKernelsFromFile(
    const std::string& path, const std::string& kernel_name,
    const std::string& extra_build_options) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) {
    return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                      "No CUDA devices available");
  }

  std::filesystem::path source_path(path);
  std::string source;
  std::string error;
  if (!ReadFile(source_path, &source, &error)) {
    return CudaStatus(StatusCode::kInvalidArgument, BackendErrorKind::kIo, error);
  }

  bool source_is_binary = IsBinarySource(source_path);

  std::vector<CudaKernel> out;
  std::string last_error;
  for (size_t i = 0; i < devices_.size(); ++i) {
    auto& dev = devices_[i];
    std::string build_options = BuildOptions(dev, extra_build_options);
    std::string cache_key = CacheKey(dev, kernel_name, build_options, source);
    auto module_or =
        BuildOrLoadModule(dev, source, build_options, cache_key, source_is_binary, kernel_name);
    if (!module_or.ok()) {
      last_error = module_or.status().message;
      continue;
    }
    gpu::CUfunction func = nullptr;
    gpu::CUresult err = loader_.cuModuleGetFunction(&func, module_or.value(), kernel_name.c_str());
    if (err != gpu::CUDA_SUCCESS) {
      last_error = "cuModuleGetFunction failed: " + gpu::CudaErrorString(err, &loader_);
      continue;
    }
    CudaKernel handle;
    handle.module = module_or.value();
    handle.func = func;
    handle.device_index = static_cast<int>(i);
    handle.name = kernel_name;
    out.push_back(handle);
  }

  if (out.empty()) {
    if (last_error.empty()) last_error = "No CUDA kernels built";
    return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kBuild, last_error);
  }
  return out;
}

Status CudaBackend::ReleaseKernel(CudaKernel* kernel) const {
  if (!kernel) return Status::OK();
  kernel->func = nullptr;
  kernel->module = nullptr;
  return Status::OK();
}

Status CudaBackend::LaunchKernel(const CudaKernel& kernel, const CudaLaunchConfig& config,
                                 const std::vector<CudaKernelArg>& args) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (kernel.device_index < 0 || kernel.device_index >= static_cast<int>(devices_.size())) {
    return CudaStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                      "Invalid CUDA device index");
  }
  auto& dev = devices_[kernel.device_index];
  if (loader_.cuCtxSetCurrent) {
    loader_.cuCtxSetCurrent(dev.context);
  }

  std::vector<gpu::CUdeviceptr> device_args;
  std::vector<std::vector<uint8_t>> value_args;
  std::vector<void*> params;
  device_args.reserve(args.size());
  value_args.reserve(args.size());
  params.reserve(args.size());

  for (const auto& arg : args) {
    if (arg.kind == CudaKernelArg::Kind::kDevicePtr) {
      device_args.push_back(arg.device_ptr);
      params.push_back(&device_args.back());
    } else {
      value_args.emplace_back(reinterpret_cast<const uint8_t*>(arg.value),
                              reinterpret_cast<const uint8_t*>(arg.value) + arg.size);
      params.push_back(value_args.back().data());
    }
  }

  gpu::CUresult err = loader_.cuLaunchKernel(
      kernel.func, config.grid[0], config.grid[1], config.grid[2], config.block[0], config.block[1],
      config.block[2], static_cast<unsigned int>(config.shared_bytes), dev.stream, params.data(),
      nullptr);
  if (err != gpu::CUDA_SUCCESS) {
    return CudaStatus(StatusCode::kInternal, BackendErrorKind::kLaunch,
                      "cuLaunchKernel failed: " + gpu::CudaErrorString(err, &loader_));
  }
  if (loader_.cuStreamSynchronize) {
    err = loader_.cuStreamSynchronize(dev.stream);
    if (err != gpu::CUDA_SUCCESS) {
      return CudaStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                        "cuStreamSynchronize failed: " + gpu::CudaErrorString(err, &loader_));
    }
  }
  return Status::OK();
}

Status CudaBackend::SmokeTest() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;

  std::string kernel_dir = KernelDir();
  if (kernel_dir.empty()) {
    return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kIo,
                      "CUDA kernel directory not found");
  }

  const std::string kernel_path = (std::filesystem::path(kernel_dir) / "lattice_smoke.cu").string();
  auto kernels_or = BuildKernelsFromFile(kernel_path, "vec_add");
  if (!kernels_or.ok()) return kernels_or.status();
  auto kernels = kernels_or.value();
  if (kernels.empty()) {
    return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                      "No CUDA devices available");
  }

  constexpr size_t kCount = 1024;
  std::vector<float> a(kCount, 1.25f);
  std::vector<float> b(kCount, 2.5f);
  std::vector<float> out(kCount, 0.0f);

  for (const auto& kernel : kernels) {
    auto buf_a_or = CreateBuffer(kernel.device_index, kCount * sizeof(float));
    if (!buf_a_or.ok()) return buf_a_or.status();
    auto buf_b_or = CreateBuffer(kernel.device_index, kCount * sizeof(float));
    if (!buf_b_or.ok()) return buf_b_or.status();
    auto buf_out_or = CreateBuffer(kernel.device_index, kCount * sizeof(float));
    if (!buf_out_or.ok()) return buf_out_or.status();

    auto buf_a = buf_a_or.value();
    auto buf_b = buf_b_or.value();
    auto buf_out = buf_out_or.value();

    status = WriteBuffer(kernel.device_index, buf_a, a.data(), a.size() * sizeof(float));
    if (!status.ok()) return status;
    status = WriteBuffer(kernel.device_index, buf_b, b.data(), b.size() * sizeof(float));
    if (!status.ok()) return status;

    CudaLaunchConfig cfg;
    cfg.block[0] = 256;
    cfg.grid[0] = static_cast<uint32_t>((kCount + cfg.block[0] - 1) / cfg.block[0]);

    unsigned int count = static_cast<unsigned int>(kCount);
    std::vector<CudaKernelArg> args;
    args.push_back(CudaKernelArg::Device(buf_a.ptr));
    args.push_back(CudaKernelArg::Device(buf_b.ptr));
    args.push_back(CudaKernelArg::Device(buf_out.ptr));
    args.push_back(CudaKernelArg::Value(&count, sizeof(count)));

    status = LaunchKernel(kernel, cfg, args);
    if (!status.ok()) return status;

    status = ReadBuffer(kernel.device_index, buf_out, out.data(), out.size() * sizeof(float));
    if (!status.ok()) return status;

    for (size_t i = 0; i < kCount; ++i) {
      if (out[i] != a[i] + b[i]) {
        return CudaStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                          "CUDA smoke test failed: incorrect output");
      }
    }

    ReleaseBuffer(&buf_a);
    ReleaseBuffer(&buf_b);
    ReleaseBuffer(&buf_out);
  }

  return Status::OK();
}

Status CudaBackend::EnsureInitialized() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (initialized_) return init_status_;
  initialized_ = true;

  std::string error;
  if (!loader_.Load(&error)) {
    init_status_ = CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kInit, error);
    return init_status_;
  }

  if (!loader_.cuInit) {
    init_status_ =
        CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kInit, "CUDA driver missing cuInit");
    return init_status_;
  }

  gpu::CUresult err = loader_.cuInit(0);
  if (err != gpu::CUDA_SUCCESS) {
    init_status_ = CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kInit,
                              "cuInit failed: " + gpu::CudaErrorString(err, &loader_));
    return init_status_;
  }

  int driver_version = 0;
  if (loader_.cuDriverGetVersion) {
    loader_.cuDriverGetVersion(&driver_version);
  }
  const std::string driver_str = FormatCudaVersion(driver_version);

  int rtc_major = 0;
  int rtc_minor = 0;
  if (loader_.nvrtcVersion) {
    loader_.nvrtcVersion(&rtc_major, &rtc_minor);
  }
  std::string runtime_str = FormatRuntimeVersion(rtc_major, rtc_minor);
  if (runtime_str.empty()) runtime_str = driver_str;

  int count = 0;
  err = loader_.cuDeviceGetCount(&count);
  if (err != gpu::CUDA_SUCCESS || count <= 0) {
    init_status_ =
        CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery, "No CUDA devices found");
    return init_status_;
  }

  std::vector<DeviceContext> candidates;
  std::vector<DeviceIdentity> identities;
  candidates.reserve(static_cast<size_t>(count));
  identities.reserve(static_cast<size_t>(count));

  for (int i = 0; i < count; ++i) {
    DeviceContext dev;
    dev.desc.index = i;
    dev.desc.vendor = "NVIDIA";
    dev.desc.driver_version = driver_str;
    dev.desc.runtime_version = runtime_str;
    if (loader_.cuDeviceGet(&dev.device, i) != gpu::CUDA_SUCCESS) continue;
    char name[256] = {0};
    if (loader_.cuDeviceGetName) {
      loader_.cuDeviceGetName(name, sizeof(name), dev.device);
      dev.desc.name = name;
    } else {
      dev.desc.name = "CUDA Device";
    }
    if (loader_.cuDeviceComputeCapability) {
      loader_.cuDeviceComputeCapability(&dev.desc.major, &dev.desc.minor, dev.device);
    }
    if (loader_.cuDeviceTotalMem) {
      size_t total_mem = 0;
      if (loader_.cuDeviceTotalMem(&total_mem, dev.device) == gpu::CUDA_SUCCESS) {
        dev.desc.total_mem = total_mem;
      }
    }
    if (loader_.cuDeviceGetAttribute) {
      int mp = 0;
      if (loader_.cuDeviceGetAttribute(&mp, kAttrMultiprocessorCount, dev.device) ==
          gpu::CUDA_SUCCESS) {
        dev.desc.multiprocessor_count = mp;
      }
      int clock = 0;
      if (loader_.cuDeviceGetAttribute(&clock, kAttrClockRate, dev.device) == gpu::CUDA_SUCCESS) {
        dev.desc.clock_khz = clock;
      }
      int max_threads = 0;
      if (loader_.cuDeviceGetAttribute(&max_threads, kAttrMaxThreadsPerBlock, dev.device) ==
          gpu::CUDA_SUCCESS) {
        dev.caps.max_threads_per_block = static_cast<size_t>(max_threads);
        dev.caps.max_work_group_size = dev.caps.max_threads_per_block;
      }
      int shared_mem = 0;
      if (loader_.cuDeviceGetAttribute(&shared_mem, kAttrSharedMemPerBlock, dev.device) ==
          gpu::CUDA_SUCCESS) {
        dev.caps.shared_mem_bytes = static_cast<size_t>(shared_mem);
      }
    }

    dev.caps.is_gpu = true;
    if (dev.desc.major == 0 && dev.desc.minor == 0) {
      dev.caps.fp64 = CapabilityStatus::kUnknown;
      dev.caps.fp16 = CapabilityStatus::kUnknown;
    } else {
      dev.caps.fp64 = (dev.desc.major > 1 || (dev.desc.major == 1 && dev.desc.minor >= 3))
                          ? CapabilityStatus::kYes
                          : CapabilityStatus::kNo;
      dev.caps.fp16 = (dev.desc.major > 5 || (dev.desc.major == 5 && dev.desc.minor >= 3))
                          ? CapabilityStatus::kYes
                          : CapabilityStatus::kNo;
    }
    dev.caps.quirks =
        QueryDeviceQuirks(BackendType::kCUDA, dev.desc.vendor, dev.desc.name, driver_str);
    dev.caps.is_software = (dev.caps.quirks.flags & kSoftwareEmulation) != 0;
    if (dev.caps.quirks.flags & kDisableFp16) dev.caps.fp16 = CapabilityStatus::kNo;
    if (dev.caps.quirks.flags & kDisableFp64) dev.caps.fp64 = CapabilityStatus::kNo;

    DeviceMetadata meta = BuildDeviceMetadata(dev.desc, dev.caps);
    dev.fingerprint = DeviceFingerprint(meta);

    DeviceIdentity identity;
    identity.index = i;
    identity.name = dev.desc.name;
    identity.vendor = dev.desc.vendor;
    identity.driver = dev.desc.driver_version;
    identity.kind = DeviceKind::kGPU;
    identities.push_back(identity);
    candidates.push_back(std::move(dev));
  }

  DeviceSelectionOptions selection = LoadDeviceSelectionOptions("LATTICE_CUDA");
  DeviceSelectionResult selected = SelectDevices(identities, selection);
  if (selected.indices.empty()) {
    init_status_ = CudaStatus(
        StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
        selected.diagnostics.empty() ? "No CUDA devices selected" : selected.diagnostics);
    return init_status_;
  }

  devices_.clear();
  devices_.reserve(selected.indices.size());
  for (int idx : selected.indices) {
    if (idx < 0 || idx >= static_cast<int>(candidates.size())) continue;
    DeviceContext dev = std::move(candidates[static_cast<size_t>(idx)]);
    if (dev.caps.quirks.disabled) {
      LogBackend({LogLevel::kWarn, BackendType::kCUDA, BackendErrorKind::kDiscovery,
                  "skipping device: " + dev.caps.quirks.reason, "device_skip", dev.desc.index,
                  dev.desc.name});
      continue;
    }
    if (!loader_.cuCtxCreate) continue;
    if (loader_.cuCtxCreate(&dev.context, 0, dev.device) != gpu::CUDA_SUCCESS) {
      LogBackend({LogLevel::kWarn, BackendType::kCUDA, BackendErrorKind::kContext,
                  "context init failed", "context", dev.desc.index, dev.desc.name});
      continue;
    }
    if (loader_.cuCtxSetCurrent) {
      loader_.cuCtxSetCurrent(dev.context);
    }
    if (loader_.cuStreamCreate) {
      if (loader_.cuStreamCreate(&dev.stream, 0) != gpu::CUDA_SUCCESS) {
        if (loader_.cuCtxDestroy) loader_.cuCtxDestroy(dev.context);
        LogBackend({LogLevel::kWarn, BackendType::kCUDA, BackendErrorKind::kContext,
                    "stream init failed", "stream", dev.desc.index, dev.desc.name});
        continue;
      }
    }
    devices_.push_back(std::move(dev));
  }

  if (devices_.empty()) {
    init_status_ = CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kContext,
                              "No CUDA devices initialized");
    return init_status_;
  }

  {
    DeviceMetadataStore meta_store;
    std::string meta_error;
    for (const auto& dev : devices_) {
      const DeviceMetadata meta = BuildDeviceMetadata(dev.desc, dev.caps);
      if (!meta_store.Write(meta, &meta_error)) {
        LogBackend({LogLevel::kWarn, BackendType::kCUDA, BackendErrorKind::kIo,
                    "metadata persist failed: " + meta_error, "metadata", dev.desc.index,
                    dev.desc.name});
        meta_error.clear();
      }
    }
  }

  for (size_t i = 0; i < devices_.size(); ++i) {
    const auto& dev = devices_[i];
    std::string info = dev.desc.name;
    if (dev.desc.major > 0) {
      info += " (sm_" + std::to_string(dev.desc.major) + std::to_string(dev.desc.minor) + ")";
    }
    LogBackend({LogLevel::kInfo, BackendType::kCUDA, BackendErrorKind::kDiscovery, info,
                "device_info", dev.desc.index, dev.desc.name});
  }

  init_status_ = Status::OK();
  return init_status_;
}

MemoryPool* CudaBackend::DevicePool(int device_index) const {
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) return nullptr;
  std::lock_guard<std::mutex> lock(mu_);
  auto& dev = devices_[device_index];
  if (dev.device_pool) return dev.device_pool.get();

  MemoryPoolConfig config = CudaDevicePoolConfig();
  const int idx = device_index;
  auto alloc_fn = [this, idx](size_t bytes, size_t alignment) -> StatusOr<PoolBlock> {
    (void)alignment;
    if (loader_.cuCtxSetCurrent) {
      loader_.cuCtxSetCurrent(devices_[idx].context);
    }
    gpu::CUdeviceptr ptr = 0;
    gpu::CUresult err = loader_.cuMemAlloc(&ptr, bytes);
    if (err != gpu::CUDA_SUCCESS) {
      return CudaStatus(StatusCode::kInternal, BackendErrorKind::kMemory,
                        "cuMemAlloc failed: " + gpu::CudaErrorString(err, &loader_));
    }
    PoolBlock block;
    block.key = static_cast<uintptr_t>(ptr);
    block.handle = static_cast<uintptr_t>(ptr);
    block.bytes = bytes;
    block.alignment = alignment;
    return block;
  };
  auto free_fn = [this, idx](const PoolBlock& block) -> Status {
    if (loader_.cuCtxSetCurrent) {
      loader_.cuCtxSetCurrent(devices_[idx].context);
    }
    auto ptr = static_cast<gpu::CUdeviceptr>(block.handle);
    gpu::CUresult err = loader_.cuMemFree(ptr);
    if (err != gpu::CUDA_SUCCESS) {
      return CudaStatus(StatusCode::kInternal, BackendErrorKind::kMemory,
                        "cuMemFree failed: " + gpu::CudaErrorString(err, &loader_));
    }
    return Status::OK();
  };
  auto scrub_fn = [this, idx](const PoolBlock& block) -> Status {
    if (!loader_.cuMemsetD8) {
      return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                        "cuMemsetD8 unavailable");
    }
    if (loader_.cuCtxSetCurrent) {
      loader_.cuCtxSetCurrent(devices_[idx].context);
    }
    auto ptr = static_cast<gpu::CUdeviceptr>(block.handle);
    gpu::CUresult err = loader_.cuMemsetD8(ptr, 0, block.bytes);
    if (err != gpu::CUDA_SUCCESS) {
      return CudaStatus(StatusCode::kInternal, BackendErrorKind::kMemory,
                        "cuMemsetD8 failed: " + gpu::CudaErrorString(err, &loader_));
    }
    if (loader_.cuStreamSynchronize && devices_[idx].stream) {
      loader_.cuStreamSynchronize(devices_[idx].stream);
    }
    return Status::OK();
  };

  std::ostringstream label;
  label << "cuda_device_pool_" << dev.desc.index;
  dev.device_pool = std::make_unique<MemoryPool>(label.str(), config, alloc_fn, free_fn, scrub_fn);
  return dev.device_pool.get();
}

MemoryPool* CudaBackend::PinnedPool() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (pinned_pool_) return pinned_pool_.get();

  MemoryPoolConfig config = CudaPinnedPoolConfig();
  auto alloc_fn = [this](size_t bytes, size_t alignment) -> StatusOr<PoolBlock> {
    (void)alignment;
    if (!loader_.cuMemHostAlloc) {
      return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                        "cuMemHostAlloc unavailable");
    }
    void* host = nullptr;
    unsigned int flags = kCuMemHostAllocPortable;
    gpu::CUresult err = loader_.cuMemHostAlloc(&host, bytes, flags);
    if (err != gpu::CUDA_SUCCESS || !host) {
      return CudaStatus(StatusCode::kInternal, BackendErrorKind::kMemory,
                        "cuMemHostAlloc failed: " + gpu::CudaErrorString(err, &loader_));
    }
    PoolBlock block;
    block.key = reinterpret_cast<uintptr_t>(host);
    block.handle = reinterpret_cast<uintptr_t>(host);
    block.host_ptr = host;
    block.bytes = bytes;
    block.alignment = alignment;
    return block;
  };
  auto free_fn = [this](const PoolBlock& block) -> Status {
    if (!loader_.cuMemFreeHost) {
      return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                        "cuMemFreeHost unavailable");
    }
    void* host = reinterpret_cast<void*>(block.handle);
    gpu::CUresult err = loader_.cuMemFreeHost(host);
    if (err != gpu::CUDA_SUCCESS) {
      return CudaStatus(StatusCode::kInternal, BackendErrorKind::kMemory,
                        "cuMemFreeHost failed: " + gpu::CudaErrorString(err, &loader_));
    }
    return Status::OK();
  };
  auto scrub_fn = [](const PoolBlock& block) -> Status {
    if (block.host_ptr && block.bytes > 0) {
      std::memset(block.host_ptr, 0, block.bytes);
    }
    return Status::OK();
  };

  pinned_pool_ =
      std::make_unique<MemoryPool>("cuda_pinned_pool", config, alloc_fn, free_fn, scrub_fn);
  return pinned_pool_.get();
}

std::string CudaBackend::KernelDir() const {
  if (const char* env = std::getenv("LATTICE_KERNEL_DIR")) {
    return std::string(env);
  }
  std::filesystem::path cwd = std::filesystem::current_path();
  for (int i = 0; i < 4; ++i) {
    std::filesystem::path candidate = cwd / "CUDA";
    if (std::filesystem::exists(candidate)) {
      return candidate.string();
    }
    cwd = cwd.parent_path();
  }
  return "";
}

std::string CudaBackend::BuildOptions(const DeviceContext& dev, const std::string& extra) const {
  std::ostringstream opts;
  opts << "--std=c++11";
  if (dev.desc.major > 0) {
    opts << " --gpu-architecture=compute_" << dev.desc.major << dev.desc.minor;
  }
  const std::string kernel_dir = KernelDir();
  if (!kernel_dir.empty()) {
    opts << " -I" << kernel_dir;
  }
  opts << " -DLATTICE_DEVICE_INDEX=" << dev.desc.index;
  opts << " -DLATTICE_ABI_VERSION=" << cuda::kAbiVersion;
  opts << " -DLATTICE_ABI_VERSION_MIN=" << cuda::kAbiVersionMin;
  if (dev.caps.fp16 == CapabilityStatus::kYes) {
    opts << " -DLATTICE_HAS_FP16=1";
  }
  if (dev.caps.fp64 == CapabilityStatus::kYes) {
    opts << " -DLATTICE_HAS_FP64=1";
  }
  if (const char* env = std::getenv("LATTICE_CUDA_BUILD_OPTIONS")) {
    if (env[0] != '\0') {
      opts << " " << env;
    }
  }
  if (!extra.empty()) {
    opts << " " << extra;
  }
  return opts.str();
}

std::string CudaBackend::CacheKey(const DeviceContext& dev, const std::string& kernel_name,
                                  const std::string& build_options,
                                  const std::string& source) const {
  std::ostringstream meta;
  meta << dev.desc.name << "|" << dev.desc.major << "." << dev.desc.minor << "|";
  meta << kernel_name << "|" << build_options;
  uint64_t meta_hash = Fnv1a64(meta.str());
  uint64_t src_hash = Fnv1a64(source);
  return "cuda_" + Hex64(meta_hash) + "_" + Hex64(src_hash);
}

StatusOr<gpu::CUmodule> CudaBackend::BuildOrLoadModule(
    DeviceContext& dev, const std::string& source, const std::string& build_options,
    const std::string& cache_key, bool source_is_binary, const std::string& kernel_name) const {
  auto it = dev.module_cache.find(cache_key);
  if (it != dev.module_cache.end()) {
    return it->second;
  }

  CacheStore& store = CudaCacheStore();
  ::lattice::runtime::CacheKey store_key{cache_key, dev.fingerprint};

  KernelTrace trace;
  trace.backend = BackendType::kCUDA;
  trace.kernel_name = kernel_name;
  trace.build_options = build_options;
  trace.source = source;
  trace.device_index = dev.desc.index;
  trace.device_name = dev.desc.name;
  std::string trace_path;
  if (TraceKernelSource(trace, &trace_path)) {
    LogBackend({LogLevel::kTrace, BackendType::kCUDA, BackendErrorKind::kBuild,
                "kernel trace written", "build", dev.desc.index, dev.desc.name, 0, "", trace_path});
  }

  auto load_module = [&](const std::string& image) -> StatusOr<gpu::CUmodule> {
    if (loader_.cuCtxSetCurrent) {
      loader_.cuCtxSetCurrent(dev.context);
    }
    gpu::CUmodule module = nullptr;
    gpu::CUresult err = gpu::CUDA_SUCCESS;
    if (loader_.cuModuleLoadDataEx) {
      err = loader_.cuModuleLoadDataEx(&module, image.data(), 0, nullptr, nullptr);
    } else {
      err = loader_.cuModuleLoadData(&module, image.data());
    }
    if (err != gpu::CUDA_SUCCESS) {
      return CudaStatus(StatusCode::kInternal, BackendErrorKind::kCompile,
                        "cuModuleLoadData failed: " + gpu::CudaErrorString(err, &loader_));
    }
    dev.module_cache.emplace(cache_key, module);
    return module;
  };

  {
    std::string cached;
    std::string error;
    if (store.ReadBinary(store_key, &cached, &error)) {
      auto module_or = load_module(cached);
      if (module_or.ok()) return module_or;
    }
  }

  std::string image;
  if (source_is_binary) {
    image = source;
  } else {
    if (!loader_.NvrtcLoaded()) {
      return CudaStatus(StatusCode::kUnavailable, BackendErrorKind::kBuild,
                        "NVRTC not available for CUDA kernel compilation");
    }
    gpu::nvrtcProgram prog = nullptr;
    gpu::NvrtcResult rc =
        loader_.nvrtcCreateProgram(&prog, source.c_str(), "lattice.cu", 0, nullptr, nullptr);
    if (rc != 0) {
      return CudaStatus(StatusCode::kInternal, BackendErrorKind::kCompile,
                        "nvrtcCreateProgram failed: " + gpu::NvrtcErrorString(rc, &loader_));
    }
    auto opts = SplitOptions(build_options);
    std::vector<const char*> opt_ptrs;
    opt_ptrs.reserve(opts.size());
    for (const auto& opt : opts) {
      opt_ptrs.push_back(opt.c_str());
    }
    rc = loader_.nvrtcCompileProgram(prog, static_cast<int>(opt_ptrs.size()),
                                     opt_ptrs.empty() ? nullptr : opt_ptrs.data());
    if (rc != 0) {
      size_t log_size = 0;
      loader_.nvrtcGetProgramLogSize(prog, &log_size);
      std::string log(log_size, '\0');
      if (log_size > 0) {
        loader_.nvrtcGetProgramLog(prog, log.data());
      }
      loader_.nvrtcDestroyProgram(&prog);
      return CudaStatus(StatusCode::kInternal, BackendErrorKind::kBuild,
                        "NVRTC compile failed: " + log);
    }
    size_t ptx_size = 0;
    rc = loader_.nvrtcGetPTXSize(prog, &ptx_size);
    if (rc != 0) {
      loader_.nvrtcDestroyProgram(&prog);
      return CudaStatus(StatusCode::kInternal, BackendErrorKind::kBuild,
                        "nvrtcGetPTXSize failed: " + gpu::NvrtcErrorString(rc, &loader_));
    }
    image.resize(ptx_size);
    rc = loader_.nvrtcGetPTX(prog, image.data());
    loader_.nvrtcDestroyProgram(&prog);
    if (rc != 0) {
      return CudaStatus(StatusCode::kInternal, BackendErrorKind::kBuild,
                        "nvrtcGetPTX failed: " + gpu::NvrtcErrorString(rc, &loader_));
    }
  }

  std::string error;
  store.WriteBinary(store_key, image.data(), image.size(), &error);

  return load_module(image);
}

const Backend* GetCudaBackend() {
  static CudaBackend* backend = [] { return new CudaBackend(); }();
  return backend;
}

Status RunCudaSmokeTest() {
  const auto* backend = static_cast<const CudaBackend*>(GetCudaBackend());
  return backend->SmokeTest();
}

}  // namespace lattice::runtime
