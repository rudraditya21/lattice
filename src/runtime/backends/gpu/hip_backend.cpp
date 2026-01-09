#include "runtime/backends/hip_backend.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/backends/backend_error.h"
#include "runtime/backends/backend_log.h"
#include "runtime/backends/cache_store.h"
#include "runtime/backends/device_quirks.h"
#include "runtime/backends/device_selector.h"
#include "runtime/backends/hip_abi.h"

namespace lattice::runtime {

namespace {

constexpr int kAttrMaxThreadsPerBlock = 1;
constexpr int kAttrSharedMemPerBlock = 8;
constexpr int kAttrClockRate = 13;
constexpr int kAttrMultiprocessorCount = 16;

std::string FormatHipVersion(int version) {
  if (version <= 0) return "";
  int major = version / 100000;
  int minor = (version / 1000) % 100;
  int patch = version % 1000;
  std::ostringstream ss;
  ss << major << "." << minor;
  if (patch != 0) {
    ss << "." << patch;
  }
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

Status HipStatus(StatusCode code, BackendErrorKind kind, const std::string& message) {
  return MakeBackendError(code, BackendType::kHIP, kind, message);
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
  return ext == ".hsaco" || ext == ".co" || ext == ".bin";
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

DeviceMetadata BuildDeviceMetadata(const HipDeviceDesc& desc, const DeviceCapabilities& caps) {
  DeviceMetadata meta;
  meta.backend = "hip";
  meta.index = desc.index;
  meta.name = desc.name;
  meta.vendor = desc.vendor;
  meta.driver_version = desc.driver_version;
  meta.runtime_version = desc.runtime_version;
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

CacheStore& HipCacheStore() {
  static CacheStore store("hip");
  return store;
}

class HipEvent final : public Event {
 public:
  void Record() override { ready_ = true; }
  void Wait() override {}
  bool Ready() const override { return ready_; }

 private:
  bool ready_ = false;
};

class HipStream final : public Stream {
 public:
  HipStream(const gpu::HipLoader* loader, gpu::hipStream_t stream)
      : loader_(loader), stream_(stream) {}

  void Submit(std::function<void()> fn) override { fn(); }

  void Synchronize() override {
    if (loader_ && loader_->hipStreamSynchronize) {
      loader_->hipStreamSynchronize(stream_);
    }
  }

  void AddDependency(const std::shared_ptr<Event>&) override {}
  void SetPriority(int) override {}

 private:
  const gpu::HipLoader* loader_ = nullptr;
  gpu::hipStream_t stream_ = nullptr;
};

}  // namespace

struct HipBackend::DeviceContext {
  gpu::hipDevice_t device = 0;
  gpu::hipCtx_t context = nullptr;
  gpu::hipStream_t stream = nullptr;
  HipDeviceDesc desc;
  DeviceCapabilities caps;
  std::string fingerprint;
  std::unordered_map<std::string, gpu::hipModule_t> module_cache;
};

HipBackend::HipBackend() = default;

HipBackend::~HipBackend() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& dev : devices_) {
    if (loader_.hipCtxSetCurrent && dev.context) {
      loader_.hipCtxSetCurrent(dev.context);
    }
    if (loader_.hipStreamDestroy && dev.stream) {
      loader_.hipStreamDestroy(dev.stream);
      dev.stream = nullptr;
    }
    if (loader_.hipCtxDestroy && dev.context) {
      loader_.hipCtxDestroy(dev.context);
      dev.context = nullptr;
    }
  }
  loader_.Unload();
}

BackendType HipBackend::Type() const {
  return BackendType::kHIP;
}

std::string HipBackend::Name() const {
  return "HIP";
}

BackendCapabilities HipBackend::Capabilities() const {
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

StatusOr<std::shared_ptr<Stream>> HipBackend::CreateStream() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) {
    return HipStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                     "No HIP devices available");
  }
  return std::make_shared<HipStream>(&loader_, devices_[0].stream);
}

StatusOr<std::shared_ptr<Event>> HipBackend::CreateEvent() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  return std::make_shared<HipEvent>();
}

StatusOr<Allocation> HipBackend::Allocate(size_t bytes, size_t alignment) const {
  (void)alignment;
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) {
    return HipStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                     "No HIP devices available");
  }
  if (loader_.hipCtxSetCurrent && devices_[0].context) {
    loader_.hipCtxSetCurrent(devices_[0].context);
  }
  void* ptr = nullptr;
  gpu::hipError_t err = loader_.hipMalloc(&ptr, bytes);
  if (err != gpu::hipSuccess) {
    return HipStatus(StatusCode::kInternal, BackendErrorKind::kMemory,
                     "hipMalloc failed: " + gpu::HipErrorString(err, &loader_));
  }
  Allocation alloc;
  alloc.ptr = nullptr;
  alloc.device_handle = ptr;
  alloc.bytes = bytes;
  alloc.alignment = alignment;
  alloc.from_pool = false;
  {
    std::lock_guard<std::mutex> lock(alloc_mu_);
    allocations_[ptr] = bytes;
  }
  return alloc;
}

Status HipBackend::Deallocate(const Allocation& alloc) const {
  if (!alloc.device_handle) return Status::OK();
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  gpu::hipError_t err = loader_.hipFree(alloc.device_handle);
  if (err != gpu::hipSuccess) {
    return HipStatus(StatusCode::kInternal, BackendErrorKind::kMemory,
                     "hipFree failed: " + gpu::HipErrorString(err, &loader_));
  }
  {
    std::lock_guard<std::mutex> lock(alloc_mu_);
    allocations_.erase(alloc.device_handle);
  }
  return Status::OK();
}

int HipBackend::NumThreads() const {
  return 1;
}

size_t HipBackend::OutstandingAllocs() const {
  std::lock_guard<std::mutex> lock(alloc_mu_);
  return allocations_.size();
}

void HipBackend::SetDefaultPriority(int priority) {
  default_priority_ = priority;
}

void HipBackend::SetDeterministic(bool deterministic) {
  deterministic_ = deterministic;
}

int HipBackend::DeviceCount() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return 0;
  return static_cast<int>(devices_.size());
}

std::vector<HipDeviceDesc> HipBackend::DeviceInfo() const {
  Status status = EnsureInitialized();
  std::vector<HipDeviceDesc> out;
  if (!status.ok()) return out;
  for (const auto& dev : devices_) {
    out.push_back(dev.desc);
  }
  return out;
}

std::vector<DeviceCapabilities> HipBackend::DeviceCaps() const {
  Status status = EnsureInitialized();
  std::vector<DeviceCapabilities> out;
  if (!status.ok()) return out;
  for (const auto& dev : devices_) {
    out.push_back(dev.caps);
  }
  return out;
}

StatusOr<HipBuffer> HipBackend::CreateBuffer(int device_index, size_t bytes) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return HipStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                     "Invalid HIP device index");
  }
  auto& dev = devices_[device_index];
  if (loader_.hipCtxSetCurrent && dev.context) {
    loader_.hipCtxSetCurrent(dev.context);
  }
  void* ptr = nullptr;
  gpu::hipError_t err = loader_.hipMalloc(&ptr, bytes);
  if (err != gpu::hipSuccess) {
    return HipStatus(StatusCode::kInternal, BackendErrorKind::kMemory,
                     "hipMalloc failed: " + gpu::HipErrorString(err, &loader_));
  }
  {
    std::lock_guard<std::mutex> lock(alloc_mu_);
    allocations_[ptr] = bytes;
  }
  return HipBuffer{ptr, bytes};
}

Status HipBackend::ReleaseBuffer(HipBuffer* buffer) const {
  if (!buffer || !buffer->ptr) return Status::OK();
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  gpu::hipError_t err = loader_.hipFree(buffer->ptr);
  if (err != gpu::hipSuccess) {
    return HipStatus(StatusCode::kInternal, BackendErrorKind::kMemory,
                     "hipFree failed: " + gpu::HipErrorString(err, &loader_));
  }
  {
    std::lock_guard<std::mutex> lock(alloc_mu_);
    allocations_.erase(buffer->ptr);
  }
  buffer->ptr = nullptr;
  buffer->bytes = 0;
  return Status::OK();
}

Status HipBackend::WriteBuffer(int device_index, const HipBuffer& buffer, const void* data,
                               size_t bytes, size_t offset) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return HipStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                     "Invalid HIP device index");
  }
  if (bytes + offset > buffer.bytes) {
    return HipStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                     "Write exceeds buffer size");
  }
  auto& dev = devices_[device_index];
  if (loader_.hipCtxSetCurrent && dev.context) {
    loader_.hipCtxSetCurrent(dev.context);
  }
  auto dst = reinterpret_cast<char*>(buffer.ptr) + static_cast<std::ptrdiff_t>(offset);
  gpu::hipError_t err = loader_.hipMemcpy(dst, data, bytes, gpu::hipMemcpyHostToDevice);
  if (err != gpu::hipSuccess) {
    return HipStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                     "hipMemcpy HtoD failed: " + gpu::HipErrorString(err, &loader_));
  }
  return Status::OK();
}

Status HipBackend::ReadBuffer(int device_index, const HipBuffer& buffer, void* data, size_t bytes,
                              size_t offset) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return HipStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                     "Invalid HIP device index");
  }
  if (bytes + offset > buffer.bytes) {
    return HipStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                     "Read exceeds buffer size");
  }
  auto& dev = devices_[device_index];
  if (loader_.hipCtxSetCurrent && dev.context) {
    loader_.hipCtxSetCurrent(dev.context);
  }
  auto src = reinterpret_cast<const char*>(buffer.ptr) + static_cast<std::ptrdiff_t>(offset);
  gpu::hipError_t err = loader_.hipMemcpy(data, src, bytes, gpu::hipMemcpyDeviceToHost);
  if (err != gpu::hipSuccess) {
    return HipStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                     "hipMemcpy DtoH failed: " + gpu::HipErrorString(err, &loader_));
  }
  return Status::OK();
}

StatusOr<HipKernel> HipBackend::BuildKernelFromFile(const std::string& path,
                                                    const std::string& kernel_name,
                                                    const std::string& extra_build_options) const {
  auto kernels_or = BuildKernelsFromFile(path, kernel_name, extra_build_options);
  if (!kernels_or.ok()) return kernels_or.status();
  auto kernels = kernels_or.value();
  if (kernels.empty()) {
    return HipStatus(StatusCode::kUnavailable, BackendErrorKind::kBuild, "No HIP kernels built");
  }
  return kernels.front();
}

StatusOr<std::vector<HipKernel>> HipBackend::BuildKernelsFromFile(
    const std::string& path, const std::string& kernel_name,
    const std::string& extra_build_options) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) {
    return HipStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                     "No HIP devices available");
  }

  std::filesystem::path source_path(path);
  std::string source;
  std::string error;
  if (!ReadFile(source_path, &source, &error)) {
    return HipStatus(StatusCode::kInvalidArgument, BackendErrorKind::kIo, error);
  }

  bool source_is_binary = IsBinarySource(source_path);

  std::vector<HipKernel> out;
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
    if (!loader_.hipModuleGetFunction) {
      last_error = "hipModuleGetFunction missing";
      continue;
    }
    gpu::hipFunction_t func = nullptr;
    gpu::hipError_t err =
        loader_.hipModuleGetFunction(&func, module_or.value(), kernel_name.c_str());
    if (err != gpu::hipSuccess) {
      last_error = "hipModuleGetFunction failed: " + gpu::HipErrorString(err, &loader_);
      continue;
    }
    HipKernel handle;
    handle.module = module_or.value();
    handle.func = func;
    handle.device_index = static_cast<int>(i);
    handle.name = kernel_name;
    out.push_back(handle);
  }

  if (out.empty()) {
    if (last_error.empty()) last_error = "No HIP kernels built";
    return HipStatus(StatusCode::kUnavailable, BackendErrorKind::kBuild, last_error);
  }
  return out;
}

Status HipBackend::ReleaseKernel(HipKernel* kernel) const {
  if (!kernel) return Status::OK();
  kernel->func = nullptr;
  kernel->module = nullptr;
  return Status::OK();
}

Status HipBackend::LaunchKernel(const HipKernel& kernel, const HipLaunchConfig& config,
                                const std::vector<HipKernelArg>& args) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (!loader_.hipModuleLaunchKernel) {
    return HipStatus(StatusCode::kUnavailable, BackendErrorKind::kUnsupported,
                     "hipModuleLaunchKernel not available");
  }
  if (kernel.device_index < 0 || kernel.device_index >= static_cast<int>(devices_.size())) {
    return HipStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                     "Invalid HIP device index");
  }
  auto& dev = devices_[kernel.device_index];
  if (loader_.hipCtxSetCurrent && dev.context) {
    loader_.hipCtxSetCurrent(dev.context);
  }

  std::vector<gpu::hipDeviceptr_t> device_args;
  std::vector<std::vector<uint8_t>> value_args;
  std::vector<void*> params;
  device_args.reserve(args.size());
  value_args.reserve(args.size());
  params.reserve(args.size());

  for (const auto& arg : args) {
    if (arg.kind == HipKernelArg::Kind::kDevicePtr) {
      device_args.push_back(arg.device_ptr);
      params.push_back(&device_args.back());
    } else {
      value_args.emplace_back(reinterpret_cast<const uint8_t*>(arg.value),
                              reinterpret_cast<const uint8_t*>(arg.value) + arg.size);
      params.push_back(value_args.back().data());
    }
  }

  gpu::hipError_t err = loader_.hipModuleLaunchKernel(
      kernel.func, config.grid[0], config.grid[1], config.grid[2], config.block[0], config.block[1],
      config.block[2], static_cast<unsigned int>(config.shared_bytes), dev.stream, params.data(),
      nullptr);
  if (err != gpu::hipSuccess) {
    return HipStatus(StatusCode::kInternal, BackendErrorKind::kLaunch,
                     "hipModuleLaunchKernel failed: " + gpu::HipErrorString(err, &loader_));
  }
  if (loader_.hipStreamSynchronize) {
    err = loader_.hipStreamSynchronize(dev.stream);
    if (err != gpu::hipSuccess) {
      return HipStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                       "hipStreamSynchronize failed: " + gpu::HipErrorString(err, &loader_));
    }
  }
  return Status::OK();
}

Status HipBackend::SmokeTest() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;

  std::string kernel_dir = KernelDir();
  if (kernel_dir.empty()) {
    return HipStatus(StatusCode::kUnavailable, BackendErrorKind::kIo,
                     "HIP kernel directory not found");
  }

  const std::string kernel_path =
      (std::filesystem::path(kernel_dir) / "lattice_smoke.hip").string();
  auto kernels_or = BuildKernelsFromFile(kernel_path, "vec_add");
  if (!kernels_or.ok()) return kernels_or.status();
  auto kernels = kernels_or.value();
  if (kernels.empty()) {
    return HipStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                     "No HIP devices available");
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

    HipLaunchConfig cfg;
    cfg.block[0] = 256;
    cfg.grid[0] = static_cast<uint32_t>((kCount + cfg.block[0] - 1) / cfg.block[0]);

    unsigned int count = static_cast<unsigned int>(kCount);
    std::vector<HipKernelArg> args;
    args.push_back(HipKernelArg::Device(buf_a.ptr));
    args.push_back(HipKernelArg::Device(buf_b.ptr));
    args.push_back(HipKernelArg::Device(buf_out.ptr));
    args.push_back(HipKernelArg::Value(&count, sizeof(count)));

    status = LaunchKernel(kernel, cfg, args);
    if (!status.ok()) return status;

    status = ReadBuffer(kernel.device_index, buf_out, out.data(), out.size() * sizeof(float));
    if (!status.ok()) return status;

    for (size_t i = 0; i < kCount; ++i) {
      if (out[i] != a[i] + b[i]) {
        return HipStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                         "HIP smoke test failed: incorrect output");
      }
    }

    ReleaseBuffer(&buf_a);
    ReleaseBuffer(&buf_b);
    ReleaseBuffer(&buf_out);
  }

  return Status::OK();
}

Status HipBackend::EnsureInitialized() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (initialized_) return init_status_;
  initialized_ = true;

  std::string error;
  if (!loader_.Load(&error)) {
    init_status_ = HipStatus(StatusCode::kUnavailable, BackendErrorKind::kInit, error);
    return init_status_;
  }

  if (loader_.hipInit) {
    loader_.hipInit(0);
  }

  int count = 0;
  gpu::hipError_t err = loader_.hipGetDeviceCount(&count);
  if (err != gpu::hipSuccess || count <= 0) {
    init_status_ =
        HipStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery, "No HIP devices found");
    return init_status_;
  }

  int driver_version = 0;
  if (loader_.hipDriverGetVersion) {
    loader_.hipDriverGetVersion(&driver_version);
  }
  const std::string driver_str = FormatHipVersion(driver_version);

  int runtime_version = 0;
  if (loader_.hipRuntimeGetVersion) {
    loader_.hipRuntimeGetVersion(&runtime_version);
  }
  std::string runtime_str = FormatHipVersion(runtime_version);
  if (runtime_str.empty()) runtime_str = driver_str;

  std::vector<DeviceContext> candidates(static_cast<size_t>(count));
  std::vector<bool> candidate_valid(static_cast<size_t>(count), false);
  std::vector<DeviceIdentity> identities;
  identities.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    DeviceContext dev;
    dev.desc.index = i;
    dev.desc.vendor = "AMD";
    dev.desc.driver_version = driver_str;
    dev.desc.runtime_version = runtime_str;
    if (loader_.hipDeviceGet(&dev.device, i) != gpu::hipSuccess) continue;
    char name[256] = {0};
    if (loader_.hipDeviceGetName) {
      loader_.hipDeviceGetName(name, sizeof(name), dev.device);
      dev.desc.name = name;
    } else {
      dev.desc.name = "HIP Device";
    }
    if (loader_.hipDeviceTotalMem) {
      size_t total_mem = 0;
      if (loader_.hipDeviceTotalMem(&total_mem, dev.device) == gpu::hipSuccess) {
        dev.desc.total_mem = total_mem;
      }
    }
    if (loader_.hipDeviceGetAttribute) {
      int max_threads = 0;
      if (loader_.hipDeviceGetAttribute(&max_threads, kAttrMaxThreadsPerBlock, dev.device) ==
          gpu::hipSuccess) {
        dev.caps.max_threads_per_block = static_cast<size_t>(max_threads);
        dev.caps.max_work_group_size = dev.caps.max_threads_per_block;
      }
      int shared_mem = 0;
      if (loader_.hipDeviceGetAttribute(&shared_mem, kAttrSharedMemPerBlock, dev.device) ==
          gpu::hipSuccess) {
        dev.caps.shared_mem_bytes = static_cast<size_t>(shared_mem);
      }
      int clock = 0;
      if (loader_.hipDeviceGetAttribute(&clock, kAttrClockRate, dev.device) == gpu::hipSuccess) {
        dev.desc.clock_khz = clock;
      }
      int mp = 0;
      if (loader_.hipDeviceGetAttribute(&mp, kAttrMultiprocessorCount, dev.device) ==
          gpu::hipSuccess) {
        dev.desc.multiprocessor_count = mp;
      }
    }

    dev.caps.is_gpu = true;
    dev.caps.fp64 = CapabilityStatus::kUnknown;
    dev.caps.fp16 = CapabilityStatus::kUnknown;
    dev.caps.quirks =
        QueryDeviceQuirks(BackendType::kHIP, dev.desc.vendor, dev.desc.name, driver_str);
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
    candidates[static_cast<size_t>(i)] = std::move(dev);
    candidate_valid[static_cast<size_t>(i)] = true;
  }

  DeviceSelectionOptions selection = LoadDeviceSelectionOptions("LATTICE_HIP");
  DeviceSelectionResult selected = SelectDevices(identities, selection);
  if (selected.indices.empty()) {
    init_status_ =
        HipStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                  selected.diagnostics.empty() ? "No HIP devices selected" : selected.diagnostics);
    return init_status_;
  }

  devices_.clear();
  devices_.reserve(selected.indices.size());
  for (int idx : selected.indices) {
    if (idx < 0 || idx >= static_cast<int>(candidates.size())) continue;
    if (!candidate_valid[static_cast<size_t>(idx)]) continue;
    DeviceContext dev = std::move(candidates[static_cast<size_t>(idx)]);
    if (dev.caps.quirks.disabled) {
      LogBackend({LogLevel::kWarn, BackendType::kHIP, BackendErrorKind::kDiscovery,
                  "skipping device: " + dev.caps.quirks.reason, "device_skip", dev.desc.index,
                  dev.desc.name});
      continue;
    }
    if (loader_.hipCtxCreate) {
      if (loader_.hipCtxCreate(&dev.context, 0, dev.device) != gpu::hipSuccess) {
        LogBackend({LogLevel::kWarn, BackendType::kHIP, BackendErrorKind::kContext,
                    "context init failed", "context", dev.desc.index, dev.desc.name});
        continue;
      }
      if (loader_.hipCtxSetCurrent) {
        loader_.hipCtxSetCurrent(dev.context);
      }
    }
    if (loader_.hipStreamCreate) {
      if (loader_.hipStreamCreate(&dev.stream) != gpu::hipSuccess) {
        if (loader_.hipCtxDestroy && dev.context) loader_.hipCtxDestroy(dev.context);
        LogBackend({LogLevel::kWarn, BackendType::kHIP, BackendErrorKind::kContext,
                    "stream init failed", "stream", dev.desc.index, dev.desc.name});
        continue;
      }
    }

    devices_.push_back(std::move(dev));
  }

  if (devices_.empty()) {
    init_status_ = HipStatus(StatusCode::kUnavailable, BackendErrorKind::kContext,
                             "No HIP devices initialized");
    return init_status_;
  }

  {
    DeviceMetadataStore meta_store;
    std::string meta_error;
    for (const auto& dev : devices_) {
      const DeviceMetadata meta = BuildDeviceMetadata(dev.desc, dev.caps);
      if (!meta_store.Write(meta, &meta_error)) {
        LogBackend({LogLevel::kWarn, BackendType::kHIP, BackendErrorKind::kIo,
                    "metadata persist failed: " + meta_error, "metadata", dev.desc.index,
                    dev.desc.name});
        meta_error.clear();
      }
    }
  }

  for (size_t i = 0; i < devices_.size(); ++i) {
    const auto& dev = devices_[i];
    LogBackend({LogLevel::kInfo, BackendType::kHIP, BackendErrorKind::kDiscovery, dev.desc.name,
                "device_info", dev.desc.index, dev.desc.name});
  }

  init_status_ = Status::OK();
  return init_status_;
}

std::string HipBackend::KernelDir() const {
  if (const char* env = std::getenv("LATTICE_KERNEL_DIR")) {
    return std::string(env);
  }
  std::filesystem::path cwd = std::filesystem::current_path();
  for (int i = 0; i < 4; ++i) {
    std::filesystem::path candidate = cwd / "HIP";
    if (std::filesystem::exists(candidate)) {
      return candidate.string();
    }
    cwd = cwd.parent_path();
  }
  return "";
}

std::string HipBackend::BuildOptions(const DeviceContext& dev, const std::string& extra) const {
  (void)dev;
  std::ostringstream opts;
  opts << "--std=c++11";
  const std::string kernel_dir = KernelDir();
  if (!kernel_dir.empty()) {
    opts << " -I" << kernel_dir;
  }
  opts << " -DLATTICE_DEVICE_INDEX=" << dev.desc.index;
  opts << " -DLATTICE_ABI_VERSION=" << hip::kAbiVersion;
  opts << " -DLATTICE_ABI_VERSION_MIN=" << hip::kAbiVersionMin;
  if (dev.caps.fp16 == CapabilityStatus::kYes) {
    opts << " -DLATTICE_HAS_FP16=1";
  }
  if (dev.caps.fp64 == CapabilityStatus::kYes) {
    opts << " -DLATTICE_HAS_FP64=1";
  }
  if (const char* arch = std::getenv("LATTICE_HIP_ARCH")) {
    if (arch[0] != '\0') {
      opts << " --gpu-architecture=" << arch;
    }
  }
  if (const char* env = std::getenv("LATTICE_HIP_BUILD_OPTIONS")) {
    if (env[0] != '\0') {
      opts << " " << env;
    }
  }
  if (!extra.empty()) {
    opts << " " << extra;
  }
  return opts.str();
}

std::string HipBackend::CacheKey(const DeviceContext& dev, const std::string& kernel_name,
                                 const std::string& build_options,
                                 const std::string& source) const {
  std::ostringstream meta;
  meta << dev.desc.name << "|" << kernel_name << "|" << build_options;
  uint64_t meta_hash = Fnv1a64(meta.str());
  uint64_t src_hash = Fnv1a64(source);
  return "hip_" + Hex64(meta_hash) + "_" + Hex64(src_hash);
}

StatusOr<gpu::hipModule_t> HipBackend::BuildOrLoadModule(
    DeviceContext& dev, const std::string& source, const std::string& build_options,
    const std::string& cache_key, bool source_is_binary, const std::string& kernel_name) const {
  auto it = dev.module_cache.find(cache_key);
  if (it != dev.module_cache.end()) {
    return it->second;
  }

  CacheStore& store = HipCacheStore();
  ::lattice::runtime::CacheKey store_key{cache_key, dev.fingerprint};

  KernelTrace trace;
  trace.backend = BackendType::kHIP;
  trace.kernel_name = kernel_name;
  trace.build_options = build_options;
  trace.source = source;
  trace.device_index = dev.desc.index;
  trace.device_name = dev.desc.name;
  std::string trace_path;
  if (TraceKernelSource(trace, &trace_path)) {
    LogBackend({LogLevel::kTrace, BackendType::kHIP, BackendErrorKind::kBuild,
                "kernel trace written", "build", dev.desc.index, dev.desc.name, 0, "", trace_path});
  }

  auto load_module = [&](const std::string& image) -> StatusOr<gpu::hipModule_t> {
    if (!loader_.hipModuleLoadData) {
      return HipStatus(StatusCode::kUnavailable, BackendErrorKind::kUnsupported,
                       "hipModuleLoadData not available");
    }
    if (loader_.hipCtxSetCurrent && dev.context) {
      loader_.hipCtxSetCurrent(dev.context);
    }
    gpu::hipModule_t module = nullptr;
    gpu::hipError_t err = loader_.hipModuleLoadData(&module, image.data());
    if (err != gpu::hipSuccess) {
      return HipStatus(StatusCode::kInternal, BackendErrorKind::kCompile,
                       "hipModuleLoadData failed: " + gpu::HipErrorString(err, &loader_));
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
    if (!loader_.HiprtcLoaded()) {
      return HipStatus(StatusCode::kUnavailable, BackendErrorKind::kBuild,
                       "HIPRTC not available for kernel compilation");
    }
    gpu::hiprtcProgram prog = nullptr;
    gpu::HiprtcResult rc =
        loader_.hiprtcCreateProgram(&prog, source.c_str(), "lattice.hip", 0, nullptr, nullptr);
    if (rc != 0) {
      return HipStatus(StatusCode::kInternal, BackendErrorKind::kCompile,
                       "hiprtcCreateProgram failed: " + gpu::HiprtcErrorString(rc, &loader_));
    }
    auto opts = SplitOptions(build_options);
    std::vector<const char*> opt_ptrs;
    opt_ptrs.reserve(opts.size());
    for (const auto& opt : opts) {
      opt_ptrs.push_back(opt.c_str());
    }
    rc = loader_.hiprtcCompileProgram(prog, static_cast<int>(opt_ptrs.size()),
                                      opt_ptrs.empty() ? nullptr : opt_ptrs.data());
    if (rc != 0) {
      size_t log_size = 0;
      loader_.hiprtcGetProgramLogSize(prog, &log_size);
      std::string log(log_size, '\0');
      if (log_size > 0) {
        loader_.hiprtcGetProgramLog(prog, log.data());
      }
      loader_.hiprtcDestroyProgram(&prog);
      return HipStatus(StatusCode::kInternal, BackendErrorKind::kBuild,
                       "HIPRTC compile failed: " + log);
    }
    size_t code_size = 0;
    rc = loader_.hiprtcGetCodeSize(prog, &code_size);
    if (rc != 0) {
      loader_.hiprtcDestroyProgram(&prog);
      return HipStatus(StatusCode::kInternal, BackendErrorKind::kBuild,
                       "hiprtcGetCodeSize failed: " + gpu::HiprtcErrorString(rc, &loader_));
    }
    image.resize(code_size);
    rc = loader_.hiprtcGetCode(prog, image.data());
    loader_.hiprtcDestroyProgram(&prog);
    if (rc != 0) {
      return HipStatus(StatusCode::kInternal, BackendErrorKind::kBuild,
                       "hiprtcGetCode failed: " + gpu::HiprtcErrorString(rc, &loader_));
    }
  }

  std::string error;
  store.WriteBinary(store_key, image.data(), image.size(), &error);

  return load_module(image);
}

const Backend* GetHipBackend() {
  static HipBackend* backend = [] { return new HipBackend(); }();
  return backend;
}

Status RunHipSmokeTest() {
  const auto* backend = static_cast<const HipBackend*>(GetHipBackend());
  return backend->SmokeTest();
}

}  // namespace lattice::runtime
