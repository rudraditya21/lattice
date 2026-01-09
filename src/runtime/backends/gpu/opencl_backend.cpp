#include "runtime/backends/opencl_backend.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/backends/cache_store.h"
#include "runtime/backends/device_quirks.h"
#include "runtime/backends/device_selector.h"
#include "runtime/backends/gpu/opencl_loader.h"
#include "runtime/backends/opencl_abi.h"

namespace lattice::runtime {

namespace {

using gpu::OpenCLLoader;

constexpr size_t kMaxPlatforms = 16;

bool HasExtension(const std::string& extensions, const char* needle) {
  return extensions.find(needle) != std::string::npos;
}

DeviceKind DeviceKindFromType(cl_device_type type) {
  if (type & CL_DEVICE_TYPE_CPU) return DeviceKind::kCPU;
  if (type & CL_DEVICE_TYPE_GPU) return DeviceKind::kGPU;
  if (type & CL_DEVICE_TYPE_ACCELERATOR) return DeviceKind::kAccelerator;
  return DeviceKind::kAny;
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

std::string NormalizePathArg(const std::filesystem::path& path) {
  const std::string raw = path.string();
  if (raw.find(' ') == std::string::npos) return raw;
  return "\"" + raw + "\"";
}

bool IsTrueEnv(const char* name) {
  const char* value = std::getenv(name);
  if (!value) return false;
  std::string v(value);
  for (char& c : v) c = static_cast<char>(std::tolower(c));
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

std::string DeviceTypeString(cl_device_type type) {
  if (type & CL_DEVICE_TYPE_GPU) return "GPU";
  if (type & CL_DEVICE_TYPE_CPU) return "CPU";
  if (type & CL_DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
  return "UNKNOWN";
}

DeviceMetadata BuildDeviceMetadata(const OpenCLDeviceDesc& desc, const DeviceCapabilities& caps) {
  DeviceMetadata meta;
  meta.backend = "opencl";
  meta.index = desc.index;
  meta.name = desc.name;
  meta.vendor = desc.vendor;
  meta.driver_version = desc.driver_version;
  meta.runtime_version = desc.runtime_version;
  meta.device_version = desc.device_version;
  meta.platform_name = desc.platform_name;
  meta.platform_vendor = desc.platform_vendor;
  meta.platform_version = desc.platform_version;
  meta.is_cpu = caps.is_cpu;
  meta.is_gpu = caps.is_gpu;
  meta.is_accel = (desc.type & CL_DEVICE_TYPE_ACCELERATOR) != 0;
  meta.fp16 = CapabilityToInt(caps.fp16);
  meta.fp64 = CapabilityToInt(caps.fp64);
  return meta;
}

CacheStore& OpenclCacheStore() {
  static CacheStore store("opencl");
  return store;
}

std::string OpenclCStd(const std::string& version) {
  // Expected format: "OpenCL C X.Y".
  auto pos = version.find("OpenCL C ");
  if (pos == std::string::npos) return "";
  std::string v = version.substr(pos + 9);
  if (v.rfind("3.0", 0) == 0) return "-cl-std=CL3.0";
  if (v.rfind("2.2", 0) == 0) return "-cl-std=CL2.2";
  if (v.rfind("2.1", 0) == 0) return "-cl-std=CL2.1";
  if (v.rfind("2.0", 0) == 0) return "-cl-std=CL2.0";
  if (v.rfind("1.2", 0) == 0) return "-cl-std=CL1.2";
  if (v.rfind("1.1", 0) == 0) return "-cl-std=CL1.1";
  return "";
}

class OpenCLEvent final : public Event {
 public:
  void Record() override {
    ready_ = std::make_shared<std::promise<void>>();
    future_ = ready_->get_future();
    ready_->set_value();
  }
  void Wait() override {
    if (future_.valid()) future_.wait();
  }
  bool Ready() const override {
    return future_.valid() &&
           future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  }

 private:
  std::shared_ptr<std::promise<void>> ready_;
  std::future<void> future_;
};

class OpenCLStream final : public Stream {
 public:
  explicit OpenCLStream(const OpenCLLoader* loader, cl_command_queue queue)
      : loader_(loader), queue_(queue) {}
  void Submit(std::function<void()> fn) override { fn(); }
  void Synchronize() override {
    if (loader_ && queue_) loader_->clFinish(queue_);
  }
  void AddDependency(const std::shared_ptr<Event>& ev) override {
    if (ev) ev->Wait();
  }
  void SetPriority(int priority) override { priority_ = priority; }

 private:
  const OpenCLLoader* loader_ = nullptr;
  cl_command_queue queue_ = nullptr;
  int priority_ = 0;
};

}  // namespace

struct OpenCLBackend::DeviceContext {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  OpenCLDeviceDesc desc;
  DeviceCapabilities caps;
  std::string fingerprint;
  std::unordered_map<std::string, cl_program> program_cache;
};

OpenCLBackend::OpenCLBackend() = default;

OpenCLBackend::~OpenCLBackend() {
  if (loader_.Loaded()) {
    for (auto& dev : devices_) {
      for (auto& entry : dev.program_cache) {
        if (entry.second) loader_.clReleaseProgram(entry.second);
      }
      dev.program_cache.clear();
      if (dev.queue) loader_.clReleaseCommandQueue(dev.queue);
      if (dev.context) loader_.clReleaseContext(dev.context);
    }
    loader_.Unload();
  }
}

BackendType OpenCLBackend::Type() const {
  return BackendType::kOpenCL;
}

std::string OpenCLBackend::Name() const {
  return "OpenCL";
}

BackendCapabilities OpenCLBackend::Capabilities() const {
  BackendCapabilities caps;
  caps.supports_dense = true;
  caps.supports_sparse = false;
  caps.supports_ragged = false;
  caps.supports_fft = false;
  caps.supports_blas = false;
  caps.supports_conv = false;
  caps.supports_rng = false;
  caps.supports_events = true;
  caps.supported_dtypes = {DType::kF16, DType::kBF16, DType::kF32,
                           DType::kF64, DType::kI32,  DType::kU32};
  return caps;
}

StatusOr<std::shared_ptr<Stream>> OpenCLBackend::CreateStream() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) return Status::Unavailable("No OpenCL devices available");
  return std::make_shared<OpenCLStream>(&loader_, devices_[0].queue);
}

StatusOr<std::shared_ptr<Event>> OpenCLBackend::CreateEvent() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  return std::make_shared<OpenCLEvent>();
}

StatusOr<Allocation> OpenCLBackend::Allocate(size_t bytes, size_t alignment) const {
  (void)alignment;
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) return Status::Unavailable("No OpenCL devices available");
  cl_int err = CL_SUCCESS;
  cl_mem buf = loader_.clCreateBuffer(devices_[0].context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
  if (err != CL_SUCCESS) {
    return Status::Internal("clCreateBuffer failed: " + gpu::OpenCLErrorString(err));
  }
  Allocation alloc;
  alloc.ptr = nullptr;
  alloc.device_handle = reinterpret_cast<void*>(buf);
  alloc.bytes = bytes;
  alloc.alignment = alignment;
  alloc.from_pool = false;
  {
    std::lock_guard<std::mutex> lock(alloc_mu_);
    allocations_[buf] = bytes;
  }
  return alloc;
}

Status OpenCLBackend::Deallocate(const Allocation& alloc) const {
  if (!alloc.device_handle) return Status::OK();
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  cl_mem buf = reinterpret_cast<cl_mem>(alloc.device_handle);
  cl_int err = loader_.clReleaseMemObject(buf);
  if (err != CL_SUCCESS) {
    return Status::Internal("clReleaseMemObject failed: " + gpu::OpenCLErrorString(err));
  }
  {
    std::lock_guard<std::mutex> lock(alloc_mu_);
    allocations_.erase(buf);
  }
  return Status::OK();
}

int OpenCLBackend::NumThreads() const {
  return 1;
}

size_t OpenCLBackend::OutstandingAllocs() const {
  std::lock_guard<std::mutex> lock(alloc_mu_);
  return allocations_.size();
}

void OpenCLBackend::SetDefaultPriority(int priority) {
  default_priority_ = priority;
}

void OpenCLBackend::SetDeterministic(bool deterministic) {
  deterministic_ = deterministic;
}

int OpenCLBackend::DeviceCount() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return 0;
  return static_cast<int>(devices_.size());
}

std::vector<OpenCLDeviceDesc> OpenCLBackend::DeviceInfo() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return {};
  std::vector<OpenCLDeviceDesc> out;
  for (const auto& dev : devices_) {
    out.push_back(dev.desc);
  }
  return out;
}

std::vector<DeviceCapabilities> OpenCLBackend::DeviceCaps() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return {};
  std::vector<DeviceCapabilities> out;
  for (const auto& dev : devices_) {
    out.push_back(dev.caps);
  }
  return out;
}

StatusOr<OpenCLBuffer> OpenCLBackend::CreateBuffer(int device_index, size_t bytes,
                                                   cl_mem_flags flags) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return Status::Invalid("Invalid device index");
  }
  cl_int err = CL_SUCCESS;
  cl_mem buf = loader_.clCreateBuffer(devices_[device_index].context, flags, bytes, nullptr, &err);
  if (err != CL_SUCCESS) {
    return Status::Internal("clCreateBuffer failed: " + gpu::OpenCLErrorString(err));
  }
  {
    std::lock_guard<std::mutex> lock(alloc_mu_);
    allocations_[buf] = bytes;
  }
  return OpenCLBuffer{buf, bytes};
}

Status OpenCLBackend::ReleaseBuffer(OpenCLBuffer* buffer) const {
  if (!buffer || !buffer->mem) return Status::OK();
  cl_int err = loader_.clReleaseMemObject(buffer->mem);
  if (err != CL_SUCCESS) {
    return Status::Internal("clReleaseMemObject failed: " + gpu::OpenCLErrorString(err));
  }
  {
    std::lock_guard<std::mutex> lock(alloc_mu_);
    allocations_.erase(buffer->mem);
  }
  buffer->mem = nullptr;
  buffer->bytes = 0;
  return Status::OK();
}

Status OpenCLBackend::WriteBuffer(int device_index, const OpenCLBuffer& buffer, const void* data,
                                  size_t bytes, size_t offset) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return Status::Invalid("Invalid device index");
  }
  if (!buffer.mem) return Status::Invalid("Invalid buffer handle");
  cl_int err = loader_.clEnqueueWriteBuffer(devices_[device_index].queue, buffer.mem, CL_TRUE,
                                            offset, bytes, data, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    return Status::Internal("clEnqueueWriteBuffer failed: " + gpu::OpenCLErrorString(err));
  }
  return Status::OK();
}

Status OpenCLBackend::ReadBuffer(int device_index, const OpenCLBuffer& buffer, void* data,
                                 size_t bytes, size_t offset) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return Status::Invalid("Invalid device index");
  }
  if (!buffer.mem) return Status::Invalid("Invalid buffer handle");
  cl_int err = loader_.clEnqueueReadBuffer(devices_[device_index].queue, buffer.mem, CL_TRUE,
                                           offset, bytes, data, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    return Status::Internal("clEnqueueReadBuffer failed: " + gpu::OpenCLErrorString(err));
  }
  return Status::OK();
}

StatusOr<OpenCLKernel> OpenCLBackend::BuildKernelFromFile(
    const std::string& path, const std::string& kernel_name,
    const std::string& extra_build_options) const {
  auto kernels_or = BuildKernelsFromFile(path, kernel_name, extra_build_options);
  if (!kernels_or.ok()) return kernels_or.status();
  auto kernels = kernels_or.value();
  if (kernels.empty()) return Status::Unavailable("No kernels built");
  return kernels.front();
}

StatusOr<std::vector<OpenCLKernel>> OpenCLBackend::BuildKernelsFromFile(
    const std::string& path, const std::string& kernel_name,
    const std::string& extra_build_options) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) return Status::Unavailable("No OpenCL devices available");

  std::filesystem::path src_path(path);
  if (!std::filesystem::exists(src_path)) {
    auto kernel_dir = KernelDir();
    if (!kernel_dir.empty()) {
      src_path = std::filesystem::path(kernel_dir) / path;
    }
  }
  if (!std::filesystem::exists(src_path)) {
    return Status::Invalid("Kernel source not found: " + src_path.string());
  }

  std::string source;
  std::string error;
  if (!ReadFile(src_path, &source, &error)) {
    return Status::Unavailable(error);
  }

  std::vector<OpenCLKernel> out;
  std::string last_error;
  for (int i = 0; i < static_cast<int>(devices_.size()); ++i) {
    auto& dev = devices_[i];
    const std::string build_options = BuildOptions(dev, extra_build_options);
    const std::string cache_key = CacheKey(dev, kernel_name, build_options, source);

    auto program_or = BuildOrLoadProgram(dev, source, build_options, cache_key);
    if (!program_or.ok()) {
      last_error = program_or.status().message;
      if (IsTrueEnv("LATTICE_OPENCL_VERBOSE")) {
        std::cerr << "* Device #" << (i + 1) << ": build failed: " << last_error << "\n";
      }
      continue;
    }
    cl_program program = program_or.value();

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = loader_.clCreateKernel(program, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS) {
      dev.program_cache.erase(cache_key);
      loader_.clReleaseProgram(program);
      last_error = "clCreateKernel failed: " + gpu::OpenCLErrorString(err);
      if (IsTrueEnv("LATTICE_OPENCL_VERBOSE")) {
        std::cerr << "* Device #" << (i + 1) << ": " << last_error << "\n";
      }
      continue;
    }

    OpenCLKernel handle;
    handle.program = program;
    handle.kernel = kernel;
    handle.device_index = i;
    handle.name = kernel_name;
    out.push_back(handle);
  }

  if (out.empty()) {
    if (last_error.empty()) last_error = "No OpenCL kernels built";
    return Status::Unavailable(last_error);
  }
  return out;
}

Status OpenCLBackend::ReleaseKernel(OpenCLKernel* kernel) const {
  if (!kernel || !kernel->kernel) return Status::OK();
  loader_.clReleaseKernel(kernel->kernel);
  kernel->kernel = nullptr;
  kernel->program = nullptr;
  return Status::OK();
}

Status OpenCLBackend::LaunchKernel(const OpenCLKernel& kernel, const OpenCLLaunchConfig& config,
                                   const std::vector<OpenCLKernelArg>& args) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (kernel.device_index < 0 || kernel.device_index >= static_cast<int>(devices_.size())) {
    return Status::Invalid("Invalid device index for kernel");
  }
  if (!kernel.kernel) return Status::Invalid("Invalid kernel handle");

  cl_int err = CL_SUCCESS;
  for (cl_uint i = 0; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg.kind == OpenCLKernelArg::Kind::kMem) {
      cl_mem mem = arg.mem;
      err = loader_.clSetKernelArg(kernel.kernel, i, sizeof(cl_mem), &mem);
    } else {
      err = loader_.clSetKernelArg(kernel.kernel, i, arg.size, arg.value);
    }
    if (err != CL_SUCCESS) {
      return Status::Internal("clSetKernelArg failed: " + gpu::OpenCLErrorString(err));
    }
  }

  const size_t* local = config.use_local ? config.local : nullptr;
  err = loader_.clEnqueueNDRangeKernel(devices_[kernel.device_index].queue, kernel.kernel,
                                       config.dims, nullptr, config.global, local, 0, nullptr,
                                       nullptr);
  if (err != CL_SUCCESS) {
    return Status::Internal("clEnqueueNDRangeKernel failed: " + gpu::OpenCLErrorString(err));
  }
  err = loader_.clFinish(devices_[kernel.device_index].queue);
  if (err != CL_SUCCESS) {
    return Status::Internal("clFinish failed: " + gpu::OpenCLErrorString(err));
  }
  return Status::OK();
}

Status OpenCLBackend::SmokeTest() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;

  const std::string kernel_dir = KernelDir();
  if (kernel_dir.empty()) {
    return Status::Unavailable("OpenCL kernel directory not found");
  }

  auto kernels_or = BuildKernelsFromFile("lattice_smoke.cl", "vec_add", "");
  if (!kernels_or.ok()) return kernels_or.status();
  auto kernels = kernels_or.value();
  if (kernels.empty()) return Status::Unavailable("No OpenCL devices available");

  const size_t n = 1024;
  std::vector<float> a(n, 1.0f);
  std::vector<float> b(n, 2.0f);
  std::vector<float> out(n, 0.0f);

  for (const auto& kernel : kernels) {
    const int device_index = kernel.device_index;
    auto buf_a_or = CreateBuffer(device_index, n * sizeof(float), CL_MEM_READ_ONLY);
    if (!buf_a_or.ok()) return buf_a_or.status();
    auto buf_b_or = CreateBuffer(device_index, n * sizeof(float), CL_MEM_READ_ONLY);
    if (!buf_b_or.ok()) return buf_b_or.status();
    auto buf_out_or = CreateBuffer(device_index, n * sizeof(float), CL_MEM_WRITE_ONLY);
    if (!buf_out_or.ok()) return buf_out_or.status();

    auto buf_a = buf_a_or.value();
    auto buf_b = buf_b_or.value();
    auto buf_out = buf_out_or.value();

    Status write_a = WriteBuffer(device_index, buf_a, a.data(), n * sizeof(float));
    if (!write_a.ok()) {
      ReleaseBuffer(&buf_a);
      ReleaseBuffer(&buf_b);
      ReleaseBuffer(&buf_out);
      return write_a;
    }
    Status write_b = WriteBuffer(device_index, buf_b, b.data(), n * sizeof(float));
    if (!write_b.ok()) {
      ReleaseBuffer(&buf_a);
      ReleaseBuffer(&buf_b);
      ReleaseBuffer(&buf_out);
      return write_b;
    }

    OpenCLLaunchConfig cfg;
    cfg.dims = 1;
    cfg.global[0] = n;

    const cl_uint count = static_cast<cl_uint>(n);
    std::vector<OpenCLKernelArg> args;
    args.push_back(OpenCLKernelArg::Mem(buf_a.mem));
    args.push_back(OpenCLKernelArg::Mem(buf_b.mem));
    args.push_back(OpenCLKernelArg::Mem(buf_out.mem));
    args.push_back(OpenCLKernelArg::Value(&count, sizeof(count)));

    Status launch = LaunchKernel(kernel, cfg, args);
    if (!launch.ok()) {
      ReleaseBuffer(&buf_a);
      ReleaseBuffer(&buf_b);
      ReleaseBuffer(&buf_out);
      return launch;
    }

    Status read = ReadBuffer(device_index, buf_out, out.data(), n * sizeof(float));
    if (!read.ok()) {
      ReleaseBuffer(&buf_a);
      ReleaseBuffer(&buf_b);
      ReleaseBuffer(&buf_out);
      return read;
    }

    const float expected = 3.0f;
    for (size_t i = 0; i < n; ++i) {
      if (std::fabs(out[i] - expected) > 1e-3f) {
        return Status::Internal("OpenCL smoke test failed: incorrect output");
      }
    }

    ReleaseBuffer(&buf_a);
    ReleaseBuffer(&buf_b);
    ReleaseBuffer(&buf_out);
  }

  for (auto& kernel : kernels) {
    ReleaseKernel(&kernel);
  }

  return Status::OK();
}

Status OpenCLBackend::EnsureInitialized() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (initialized_) return init_status_;
  initialized_ = true;

  std::string error;
  if (!loader_.Load(&error)) {
    init_status_ = Status::Unavailable(error);
    return init_status_;
  }

  cl_uint num_platforms = 0;
  cl_int err = loader_.clGetPlatformIDs(0, nullptr, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    init_status_ = Status::Unavailable("No OpenCL platforms found");
    return init_status_;
  }

  num_platforms = std::min<cl_uint>(num_platforms, kMaxPlatforms);
  std::vector<cl_platform_id> platforms(num_platforms);
  err = loader_.clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (err != CL_SUCCESS) {
    init_status_ = Status::Unavailable("clGetPlatformIDs failed: " + gpu::OpenCLErrorString(err));
    return init_status_;
  }

  std::vector<DeviceContext> candidates;
  std::vector<DeviceIdentity> identities;
  int device_index = 0;
  for (cl_platform_id platform : platforms) {
    cl_uint num_devices = 0;
    err = loader_.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
    if (err == CL_DEVICE_NOT_FOUND || num_devices == 0) {
      continue;
    }
    if (err != CL_SUCCESS) {
      continue;
    }
    std::vector<cl_device_id> devices(num_devices);
    err =
        loader_.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
      continue;
    }

    const std::string platform_name = PlatformInfoString(platform, CL_PLATFORM_NAME);
    const std::string platform_vendor = PlatformInfoString(platform, CL_PLATFORM_VENDOR);
    const std::string platform_version = PlatformInfoString(platform, CL_PLATFORM_VERSION);

    for (cl_device_id device : devices) {
      DeviceContext ctx;
      ctx.platform = platform;
      ctx.device = device;
      ctx.desc.index = device_index;
      ctx.desc.platform_name = platform_name;
      ctx.desc.platform_vendor = platform_vendor;
      ctx.desc.platform_version = platform_version;
      ctx.desc.runtime_version = platform_version;

      ctx.desc.name = DeviceInfoString(device, CL_DEVICE_NAME);
      ctx.desc.vendor = DeviceInfoString(device, CL_DEVICE_VENDOR);
      ctx.desc.device_version = DeviceInfoString(device, CL_DEVICE_VERSION);
      ctx.desc.driver_version = DeviceInfoString(device, CL_DRIVER_VERSION);
      loader_.clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(ctx.desc.type), &ctx.desc.type,
                              nullptr);
      loader_.clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(ctx.desc.vendor_id),
                              &ctx.desc.vendor_id, nullptr);
      loader_.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                              sizeof(ctx.desc.max_work_group_size), &ctx.desc.max_work_group_size,
                              nullptr);
      loader_.clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ctx.desc.local_mem_size),
                              &ctx.desc.local_mem_size, nullptr);
      loader_.clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(ctx.desc.local_mem_type),
                              &ctx.desc.local_mem_type, nullptr);
      loader_.clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ctx.desc.compute_units),
                              &ctx.desc.compute_units, nullptr);
      loader_.clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(ctx.desc.max_clock_mhz),
                              &ctx.desc.max_clock_mhz, nullptr);
      size_t work_item_sizes[3] = {0, 0, 0};
      loader_.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(work_item_sizes),
                              work_item_sizes, nullptr);

      const std::string extensions = DeviceInfoString(device, CL_DEVICE_EXTENSIONS);
      ctx.caps.is_cpu = (ctx.desc.type & CL_DEVICE_TYPE_CPU) != 0;
      ctx.caps.is_gpu = (ctx.desc.type & CL_DEVICE_TYPE_GPU) != 0;
      ctx.caps.local_mem_bytes = ctx.desc.local_mem_size;
      ctx.caps.max_work_group_size = ctx.desc.max_work_group_size;
      ctx.caps.max_threads_per_block = ctx.desc.max_work_group_size;
      ctx.caps.max_work_item_sizes[0] = work_item_sizes[0];
      ctx.caps.max_work_item_sizes[1] = work_item_sizes[1];
      ctx.caps.max_work_item_sizes[2] = work_item_sizes[2];
      ctx.caps.fp64 =
          HasExtension(extensions, "cl_khr_fp64") || HasExtension(extensions, "cl_amd_fp64")
              ? CapabilityStatus::kYes
              : CapabilityStatus::kNo;
      ctx.caps.fp16 =
          HasExtension(extensions, "cl_khr_fp16") || HasExtension(extensions, "cl_amd_fp16")
              ? CapabilityStatus::kYes
              : CapabilityStatus::kNo;
      ctx.caps.quirks = QueryDeviceQuirks(BackendType::kOpenCL, ctx.desc.vendor, ctx.desc.name,
                                          ctx.desc.driver_version);
      ctx.caps.is_software = (ctx.caps.quirks.flags & kSoftwareEmulation) != 0;
      if (ctx.caps.quirks.flags & kDisableFp16) {
        ctx.caps.fp16 = CapabilityStatus::kNo;
      }
      if (ctx.caps.quirks.flags & kDisableFp64) {
        ctx.caps.fp64 = CapabilityStatus::kNo;
      }

      DeviceMetadata meta = BuildDeviceMetadata(ctx.desc, ctx.caps);
      ctx.fingerprint = DeviceFingerprint(meta);

      DeviceIdentity identity;
      identity.index = device_index;
      identity.name = ctx.desc.name;
      identity.vendor = ctx.desc.vendor;
      identity.driver = ctx.desc.driver_version;
      identity.kind = DeviceKindFromType(ctx.desc.type);
      identities.push_back(identity);
      candidates.push_back(std::move(ctx));
      ++device_index;
    }
  }

  DeviceSelectionOptions selection = LoadDeviceSelectionOptions("LATTICE_OPENCL");
  DeviceSelectionResult selected = SelectDevices(identities, selection);
  if (selected.indices.empty()) {
    init_status_ = Status::Unavailable(selected.diagnostics.empty() ? "No OpenCL devices selected"
                                                                    : selected.diagnostics);
    return init_status_;
  }

  for (int idx : selected.indices) {
    if (idx < 0 || idx >= static_cast<int>(candidates.size())) continue;
    DeviceContext ctx = std::move(candidates[static_cast<size_t>(idx)]);
    if (ctx.caps.quirks.disabled) {
      if (IsTrueEnv("LATTICE_OPENCL_VERBOSE")) {
        std::cerr << "* Skipping OpenCL device '" << ctx.desc.name
                  << "': " << ctx.caps.quirks.reason << "\n";
      }
      continue;
    }
    cl_int create_err = CL_SUCCESS;
    ctx.context = loader_.clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, &create_err);
    if (create_err != CL_SUCCESS || !ctx.context) {
      if (IsTrueEnv("LATTICE_OPENCL_VERBOSE")) {
        std::cerr << "* OpenCL device '" << ctx.desc.name << "' context init failed\n";
      }
      continue;
    }
    ctx.queue = loader_.clCreateCommandQueue(ctx.context, ctx.device, 0, &create_err);
    if (create_err != CL_SUCCESS || !ctx.queue) {
      loader_.clReleaseContext(ctx.context);
      ctx.context = nullptr;
      if (IsTrueEnv("LATTICE_OPENCL_VERBOSE")) {
        std::cerr << "* OpenCL device '" << ctx.desc.name << "' queue init failed\n";
      }
      continue;
    }
    devices_.push_back(std::move(ctx));
  }

  if (devices_.empty()) {
    if (!selected.indices.empty()) {
      init_status_ =
          Status::Unavailable("No OpenCL devices initialized (all selected devices failed)");
    } else {
      init_status_ = Status::Unavailable("No OpenCL devices found");
    }
    return init_status_;
  }

  {
    DeviceMetadataStore meta_store;
    std::string meta_error;
    for (const auto& dev : devices_) {
      const DeviceMetadata meta = BuildDeviceMetadata(dev.desc, dev.caps);
      if (!meta_store.Write(meta, &meta_error) && IsTrueEnv("LATTICE_OPENCL_VERBOSE")) {
        std::cerr << "* Failed to persist OpenCL metadata: " << meta_error << "\n";
        meta_error.clear();
      }
    }
  }

  if (IsTrueEnv("LATTICE_OPENCL_VERBOSE")) {
    for (size_t i = 0; i < devices_.size(); ++i) {
      const auto& dev = devices_[i].desc;
      std::cerr << "* OpenCL Device #" << (i + 1) << ": " << dev.name << " (";
      std::cerr << DeviceTypeString(dev.type) << ") vendor=" << dev.vendor;
      std::cerr << " driver=" << dev.driver_version << "\n";
    }
  }

  init_status_ = Status::OK();
  return init_status_;
}

std::string OpenCLBackend::KernelDir() const {
  if (const char* env = std::getenv("LATTICE_KERNEL_DIR")) {
    return env;
  }
  std::filesystem::path cwd = std::filesystem::current_path();
  for (int depth = 0; depth < 4; ++depth) {
    std::filesystem::path candidate = cwd / "OpenCL";
    if (std::filesystem::exists(candidate)) {
      return candidate.string();
    }
    if (!cwd.has_parent_path()) break;
    cwd = cwd.parent_path();
  }
  return "";
}

std::string OpenCLBackend::DeviceInfoString(cl_device_id device, cl_device_info param) const {
  size_t size = 0;
  loader_.clGetDeviceInfo(device, param, 0, nullptr, &size);
  if (size == 0) return "";
  std::string out(size, '\0');
  loader_.clGetDeviceInfo(device, param, size, out.data(), nullptr);
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

std::string OpenCLBackend::PlatformInfoString(cl_platform_id platform,
                                              cl_platform_info param) const {
  size_t size = 0;
  loader_.clGetPlatformInfo(platform, param, 0, nullptr, &size);
  if (size == 0) return "";
  std::string out(size, '\0');
  loader_.clGetPlatformInfo(platform, param, size, out.data(), nullptr);
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

std::string OpenCLBackend::BuildOptions(const DeviceContext& dev, const std::string& extra) const {
  std::string options;
  const std::string kernel_dir = KernelDir();
  if (!kernel_dir.empty()) {
    options += "-I ";
    options += NormalizePathArg(kernel_dir);
  }

  if (const char* env = std::getenv("LATTICE_OPENCL_BUILD_OPTIONS")) {
    if (!options.empty()) options.push_back(' ');
    options += env;
  }

  if (!extra.empty()) {
    if (!options.empty()) options.push_back(' ');
    options += extra;
  }

  std::string c_version = DeviceInfoString(dev.device, CL_DEVICE_OPENCL_C_VERSION);
  std::string std_opt = OpenclCStd(c_version);
  if (!std_opt.empty()) {
    if (!options.empty()) options.push_back(' ');
    options += std_opt;
  }

  if (!options.empty()) options.push_back(' ');
  options += "-D LATTICE_DEVICE_TYPE=" + std::to_string(static_cast<uint64_t>(dev.desc.type));
  options += " -D LATTICE_VENDOR_ID=" + std::to_string(dev.desc.vendor_id);
  options += " -D LATTICE_DEVICE_INDEX=" + std::to_string(dev.desc.index);
  options += " -D LATTICE_ABI_VERSION=" + std::to_string(opencl::kAbiVersion);
  options += " -D LATTICE_ABI_VERSION_MIN=" + std::to_string(opencl::kAbiVersionMin);

  std::string extensions = DeviceInfoString(dev.device, CL_DEVICE_EXTENSIONS);
  if (extensions.find("cl_khr_fp64") != std::string::npos) {
    options += " -D LATTICE_HAS_FP64";
  }
  if (extensions.find("cl_khr_fp16") != std::string::npos) {
    options += " -D LATTICE_HAS_FP16";
  }

  if (IsTrueEnv("LATTICE_OPENCL_VERBOSE")) {
    std::cerr << "* Device #" << (dev.desc.index + 1) << ": build_options '" << options << "'\n";
  }

  return options;
}

std::string OpenCLBackend::CacheKey(const DeviceContext& dev, const std::string& kernel_name,
                                    const std::string& build_options,
                                    const std::string& source) const {
  std::ostringstream ss;
  ss << dev.desc.name << "|" << dev.desc.vendor << "|" << dev.desc.driver_version << "|";
  ss << build_options << "|" << kernel_name;
  const std::string meta = ss.str();
  uint64_t meta_hash = Fnv1a64(meta);
  uint64_t src_hash = Fnv1a64(source);
  return "opencl_" + Hex64(meta_hash) + "_" + Hex64(src_hash);
}

StatusOr<cl_program> OpenCLBackend::BuildOrLoadProgram(DeviceContext& dev,
                                                       const std::string& source,
                                                       const std::string& build_options,
                                                       const std::string& cache_key) const {
  auto it = dev.program_cache.find(cache_key);
  if (it != dev.program_cache.end()) {
    return it->second;
  }
  CacheStore& store = OpenclCacheStore();
  ::lattice::runtime::CacheKey store_key{cache_key, dev.fingerprint};
  std::string binary;
  std::string error;
  if (store.ReadBinary(store_key, &binary, &error)) {
    const unsigned char* bin_ptr = reinterpret_cast<const unsigned char*>(binary.data());
    size_t bin_size = binary.size();
    cl_int status = CL_SUCCESS;
    cl_program program = loader_.clCreateProgramWithBinary(dev.context, 1, &dev.device, &bin_size,
                                                           &bin_ptr, nullptr, &status);
    if (status == CL_SUCCESS && program) {
      cl_int err =
          loader_.clBuildProgram(program, 1, &dev.device, build_options.c_str(), nullptr, nullptr);
      if (err == CL_SUCCESS) {
        dev.program_cache[cache_key] = program;
        return program;
      }
      loader_.clReleaseProgram(program);
    }
  }

  const char* src = source.c_str();
  cl_int err = CL_SUCCESS;
  cl_program program = loader_.clCreateProgramWithSource(dev.context, 1, &src, nullptr, &err);
  if (err != CL_SUCCESS || !program) {
    return Status::Internal("clCreateProgramWithSource failed: " + gpu::OpenCLErrorString(err));
  }

  err = loader_.clBuildProgram(program, 1, &dev.device, build_options.c_str(), nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    loader_.clGetProgramBuildInfo(program, dev.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::string log(log_size, '\0');
    if (log_size > 1) {
      loader_.clGetProgramBuildInfo(program, dev.device, CL_PROGRAM_BUILD_LOG, log_size, log.data(),
                                    nullptr);
      if (!log.empty() && log.back() == '\0') log.pop_back();
    }
    loader_.clReleaseProgram(program);
    return Status::Internal("OpenCL build failed: " + log);
  }

  size_t bin_size = 0;
  err = loader_.clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size,
                                 nullptr);
  if (err == CL_SUCCESS && bin_size > 0) {
    std::vector<unsigned char> bin(bin_size);
    unsigned char* bin_ptr = bin.data();
    err = loader_.clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &bin_ptr,
                                   nullptr);
    if (err == CL_SUCCESS) {
      store.WriteBinary(store_key, bin.data(), bin.size(), &error);
    }
  }

  dev.program_cache[cache_key] = program;
  return program;
}

const Backend* GetOpenCLBackend() {
  static OpenCLBackend* backend = [] { return new OpenCLBackend(); }();
  return backend;
}

Status RunOpenCLSmokeTest() {
  const auto* backend = static_cast<const OpenCLBackend*>(GetOpenCLBackend());
  return backend->SmokeTest();
}

}  // namespace lattice::runtime
