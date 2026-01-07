#include "runtime/backends/cuda_backend.h"

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

namespace lattice::runtime {

namespace {

constexpr int kAttrMultiprocessorCount = 16;
constexpr int kAttrClockRate = 13;  // kHz

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

std::filesystem::path DefaultCacheDir() {
  if (const char* env = std::getenv("LATTICE_CACHE_DIR")) {
    return std::filesystem::path(env);
  }
  return std::filesystem::current_path() / ".lattice_cache";
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

bool WriteFile(const std::filesystem::path& path, const void* data, size_t size,
               std::string* error) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    if (error) {
      *error = "Failed to open file for write: " + path.string();
    }
    return false;
  }
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!out) {
    if (error) {
      *error = "Failed to write file: " + path.string();
    }
    return false;
  }
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
  std::unordered_map<std::string, gpu::CUmodule> module_cache;
};

CudaBackend::CudaBackend() = default;

CudaBackend::~CudaBackend() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& dev : devices_) {
    if (loader_.cuCtxSetCurrent && dev.context) {
      loader_.cuCtxSetCurrent(dev.context);
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
  if (devices_.empty()) return Status::Unavailable("No CUDA devices available");
  return std::make_shared<CudaStream>(&loader_, devices_[0].stream);
}

StatusOr<std::shared_ptr<Event>> CudaBackend::CreateEvent() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  return std::make_shared<CudaEvent>();
}

StatusOr<Allocation> CudaBackend::Allocate(size_t bytes, size_t alignment) const {
  (void)alignment;
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) return Status::Unavailable("No CUDA devices available");
  gpu::CUdeviceptr ptr = 0;
  if (loader_.cuCtxSetCurrent) {
    loader_.cuCtxSetCurrent(devices_[0].context);
  }
  gpu::CUresult err = loader_.cuMemAlloc(&ptr, bytes);
  if (err != gpu::CUDA_SUCCESS) {
    return Status::Internal("cuMemAlloc failed: " + gpu::CudaErrorString(err, &loader_));
  }
  Allocation alloc;
  alloc.ptr = nullptr;
  alloc.device_handle = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
  alloc.bytes = bytes;
  alloc.alignment = alignment;
  alloc.from_pool = false;
  {
    std::lock_guard<std::mutex> lock(alloc_mu_);
    allocations_[ptr] = bytes;
  }
  return alloc;
}

Status CudaBackend::Deallocate(const Allocation& alloc) const {
  if (!alloc.device_handle) return Status::OK();
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  auto ptr = static_cast<gpu::CUdeviceptr>(reinterpret_cast<uintptr_t>(alloc.device_handle));
  gpu::CUresult err = loader_.cuMemFree(ptr);
  if (err != gpu::CUDA_SUCCESS) {
    return Status::Internal("cuMemFree failed: " + gpu::CudaErrorString(err, &loader_));
  }
  {
    std::lock_guard<std::mutex> lock(alloc_mu_);
    allocations_.erase(ptr);
  }
  return Status::OK();
}

int CudaBackend::NumThreads() const {
  return 1;
}

size_t CudaBackend::OutstandingAllocs() const {
  std::lock_guard<std::mutex> lock(alloc_mu_);
  return allocations_.size();
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

StatusOr<CudaBuffer> CudaBackend::CreateBuffer(int device_index, size_t bytes) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return Status::Invalid("Invalid CUDA device index");
  }
  auto& dev = devices_[device_index];
  if (loader_.cuCtxSetCurrent) {
    loader_.cuCtxSetCurrent(dev.context);
  }
  gpu::CUdeviceptr ptr = 0;
  gpu::CUresult err = loader_.cuMemAlloc(&ptr, bytes);
  if (err != gpu::CUDA_SUCCESS) {
    return Status::Internal("cuMemAlloc failed: " + gpu::CudaErrorString(err, &loader_));
  }
  {
    std::lock_guard<std::mutex> lock(alloc_mu_);
    allocations_[ptr] = bytes;
  }
  return CudaBuffer{ptr, bytes};
}

Status CudaBackend::ReleaseBuffer(CudaBuffer* buffer) const {
  if (!buffer || !buffer->ptr) return Status::OK();
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  gpu::CUresult err = loader_.cuMemFree(buffer->ptr);
  if (err != gpu::CUDA_SUCCESS) {
    return Status::Internal("cuMemFree failed: " + gpu::CudaErrorString(err, &loader_));
  }
  {
    std::lock_guard<std::mutex> lock(alloc_mu_);
    allocations_.erase(buffer->ptr);
  }
  buffer->ptr = 0;
  buffer->bytes = 0;
  return Status::OK();
}

Status CudaBackend::WriteBuffer(int device_index, const CudaBuffer& buffer, const void* data,
                                size_t bytes, size_t offset) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return Status::Invalid("Invalid CUDA device index");
  }
  if (bytes + offset > buffer.bytes) return Status::Invalid("Write exceeds buffer size");
  auto& dev = devices_[device_index];
  if (loader_.cuCtxSetCurrent) {
    loader_.cuCtxSetCurrent(dev.context);
  }
  gpu::CUresult err = loader_.cuMemcpyHtoD(buffer.ptr + offset, data, bytes);
  if (err != gpu::CUDA_SUCCESS) {
    return Status::Internal("cuMemcpyHtoD failed: " + gpu::CudaErrorString(err, &loader_));
  }
  return Status::OK();
}

Status CudaBackend::ReadBuffer(int device_index, const CudaBuffer& buffer, void* data, size_t bytes,
                               size_t offset) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return Status::Invalid("Invalid CUDA device index");
  }
  if (bytes + offset > buffer.bytes) return Status::Invalid("Read exceeds buffer size");
  auto& dev = devices_[device_index];
  if (loader_.cuCtxSetCurrent) {
    loader_.cuCtxSetCurrent(dev.context);
  }
  gpu::CUresult err = loader_.cuMemcpyDtoH(data, buffer.ptr + offset, bytes);
  if (err != gpu::CUDA_SUCCESS) {
    return Status::Internal("cuMemcpyDtoH failed: " + gpu::CudaErrorString(err, &loader_));
  }
  return Status::OK();
}

StatusOr<CudaKernel> CudaBackend::BuildKernelFromFile(
    const std::string& path, const std::string& kernel_name,
    const std::string& extra_build_options) const {
  auto kernels_or = BuildKernelsFromFile(path, kernel_name, extra_build_options);
  if (!kernels_or.ok()) return kernels_or.status();
  auto kernels = kernels_or.value();
  if (kernels.empty()) return Status::Unavailable("No CUDA kernels built");
  return kernels.front();
}

StatusOr<std::vector<CudaKernel>> CudaBackend::BuildKernelsFromFile(
    const std::string& path, const std::string& kernel_name,
    const std::string& extra_build_options) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) return Status::Unavailable("No CUDA devices available");

  std::filesystem::path source_path(path);
  std::string source;
  std::string error;
  if (!ReadFile(source_path, &source, &error)) {
    return Status::Invalid(error);
  }

  bool source_is_binary = IsBinarySource(source_path);

  std::vector<CudaKernel> out;
  std::string last_error;
  for (size_t i = 0; i < devices_.size(); ++i) {
    auto& dev = devices_[i];
    std::string build_options = BuildOptions(dev, extra_build_options);
    std::string cache_key = CacheKey(dev, kernel_name, build_options, source);
    auto module_or = BuildOrLoadModule(dev, source, build_options, cache_key, source_is_binary);
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
    return Status::Unavailable(last_error);
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
    return Status::Invalid("Invalid CUDA device index");
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
    return Status::Internal("cuLaunchKernel failed: " + gpu::CudaErrorString(err, &loader_));
  }
  if (loader_.cuStreamSynchronize) {
    err = loader_.cuStreamSynchronize(dev.stream);
    if (err != gpu::CUDA_SUCCESS) {
      return Status::Internal("cuStreamSynchronize failed: " + gpu::CudaErrorString(err, &loader_));
    }
  }
  return Status::OK();
}

Status CudaBackend::SmokeTest() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;

  std::string kernel_dir = KernelDir();
  if (kernel_dir.empty()) return Status::Unavailable("CUDA kernel directory not found");

  const std::string kernel_path = (std::filesystem::path(kernel_dir) / "lattice_smoke.cu").string();
  auto kernels_or = BuildKernelsFromFile(kernel_path, "vec_add");
  if (!kernels_or.ok()) return kernels_or.status();
  auto kernels = kernels_or.value();
  if (kernels.empty()) return Status::Unavailable("No CUDA devices available");

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
        return Status::Internal("CUDA smoke test failed: incorrect output");
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
    init_status_ = Status::Unavailable(error);
    return init_status_;
  }

  if (!loader_.cuInit) {
    init_status_ = Status::Unavailable("CUDA driver missing cuInit");
    return init_status_;
  }

  gpu::CUresult err = loader_.cuInit(0);
  if (err != gpu::CUDA_SUCCESS) {
    init_status_ = Status::Unavailable("cuInit failed: " + gpu::CudaErrorString(err, &loader_));
    return init_status_;
  }

  int count = 0;
  err = loader_.cuDeviceGetCount(&count);
  if (err != gpu::CUDA_SUCCESS || count <= 0) {
    init_status_ = Status::Unavailable("No CUDA devices found");
    return init_status_;
  }

  devices_.clear();
  devices_.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    DeviceContext dev;
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
    }

    if (!loader_.cuCtxCreate) continue;
    if (loader_.cuCtxCreate(&dev.context, 0, dev.device) != gpu::CUDA_SUCCESS) {
      continue;
    }
    if (loader_.cuCtxSetCurrent) {
      loader_.cuCtxSetCurrent(dev.context);
    }
    if (loader_.cuStreamCreate) {
      if (loader_.cuStreamCreate(&dev.stream, 0) != gpu::CUDA_SUCCESS) {
        if (loader_.cuCtxDestroy) loader_.cuCtxDestroy(dev.context);
        continue;
      }
    }

    devices_.push_back(std::move(dev));
  }

  if (devices_.empty()) {
    init_status_ = Status::Unavailable("No CUDA devices initialized");
    return init_status_;
  }

  if (const char* verbose = std::getenv("LATTICE_CUDA_VERBOSE")) {
    if (verbose[0] != '\0') {
      for (size_t i = 0; i < devices_.size(); ++i) {
        const auto& dev = devices_[i];
        std::cerr << "* CUDA Device #" << (i + 1) << ": " << dev.desc.name;
        if (dev.desc.major > 0) {
          std::cerr << " (sm_" << dev.desc.major << dev.desc.minor << ")";
        }
        std::cerr << "\n";
      }
    }
  }

  init_status_ = Status::OK();
  return init_status_;
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

StatusOr<gpu::CUmodule> CudaBackend::BuildOrLoadModule(DeviceContext& dev,
                                                       const std::string& source,
                                                       const std::string& build_options,
                                                       const std::string& cache_key,
                                                       bool source_is_binary) const {
  auto it = dev.module_cache.find(cache_key);
  if (it != dev.module_cache.end()) {
    return it->second;
  }

  std::filesystem::path cache_dir = DefaultCacheDir();
  std::error_code ec;
  std::filesystem::create_directories(cache_dir, ec);
  std::filesystem::path cache_path = cache_dir / (cache_key + ".bin");

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
      return Status::Internal("cuModuleLoadData failed: " + gpu::CudaErrorString(err, &loader_));
    }
    dev.module_cache.emplace(cache_key, module);
    return module;
  };

  if (std::filesystem::exists(cache_path)) {
    std::string cached;
    std::string error;
    if (ReadFile(cache_path, &cached, &error)) {
      auto module_or = load_module(cached);
      if (module_or.ok()) return module_or;
    }
  }

  std::string image;
  if (source_is_binary) {
    image = source;
  } else {
    if (!loader_.NvrtcLoaded()) {
      return Status::Unavailable("NVRTC not available for CUDA kernel compilation");
    }
    gpu::nvrtcProgram prog = nullptr;
    gpu::NvrtcResult rc =
        loader_.nvrtcCreateProgram(&prog, source.c_str(), "lattice.cu", 0, nullptr, nullptr);
    if (rc != 0) {
      return Status::Internal("nvrtcCreateProgram failed: " + gpu::NvrtcErrorString(rc, &loader_));
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
      return Status::Internal("NVRTC compile failed: " + log);
    }
    size_t ptx_size = 0;
    rc = loader_.nvrtcGetPTXSize(prog, &ptx_size);
    if (rc != 0) {
      loader_.nvrtcDestroyProgram(&prog);
      return Status::Internal("nvrtcGetPTXSize failed: " + gpu::NvrtcErrorString(rc, &loader_));
    }
    image.resize(ptx_size);
    rc = loader_.nvrtcGetPTX(prog, image.data());
    loader_.nvrtcDestroyProgram(&prog);
    if (rc != 0) {
      return Status::Internal("nvrtcGetPTX failed: " + gpu::NvrtcErrorString(rc, &loader_));
    }
  }

  std::string error;
  WriteFile(cache_path, image.data(), image.size(), &error);

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
