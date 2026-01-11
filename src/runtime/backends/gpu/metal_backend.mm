#include "runtime/backends/metal_backend.h"

#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>

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
#include "runtime/backends/device_quirks.h"
#include "runtime/backends/device_selector.h"
#include "runtime/backends/memory_pool.h"
#include "runtime/backends/metal_abi.h"

namespace lattice::runtime {

namespace {

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

Status MetalStatus(StatusCode code, BackendErrorKind kind, const std::string& message) {
  return MakeBackendError(code, BackendType::kMetal, kind, message);
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

class MetalEvent final : public Event {
 public:
  void Record() override { ready_ = true; }
  void Wait() override {}
  bool Ready() const override { return ready_; }

 private:
  bool ready_ = false;
};

class MetalStream final : public Stream {
 public:
  void Submit(std::function<void()> fn) override { fn(); }
  void Synchronize() override {}
  void AddDependency(const std::shared_ptr<Event>&) override {}
  void SetPriority(int) override {}
};

bool QueryBoolSelector(id obj, SEL sel, bool* out) {
  if (![obj respondsToSelector:sel]) return false;
  BOOL value = ((BOOL (*)(id, SEL))objc_msgSend)(obj, sel);
  *out = value != NO;
  return true;
}

DeviceMetadata BuildDeviceMetadata(const MetalDeviceDesc& desc, const DeviceCapabilities& caps) {
  DeviceMetadata meta;
  meta.backend = "metal";
  meta.index = desc.index;
  meta.name = desc.name;
  meta.vendor = desc.vendor;
  meta.driver_version = desc.driver_version;
  meta.runtime_version = desc.runtime_version;
  meta.is_gpu = caps.is_gpu;
  meta.is_cpu = caps.is_cpu;
  meta.is_accel = false;
  meta.fp16 = CapabilityToInt(caps.fp16);
  meta.fp64 = CapabilityToInt(caps.fp64);
  return meta;
}

MemoryPoolConfig MetalDevicePoolConfig() {
  static MemoryPoolConfig config = [] {
    MemoryPoolConfig base = DefaultDevicePoolConfig();
    base = LoadMemoryPoolConfig("LATTICE_DEVICE_POOL", base);
    base = LoadMemoryPoolConfig("LATTICE_METAL_DEVICE_POOL", base);
    return base;
  }();
  return config;
}

MemoryPoolConfig MetalPinnedPoolConfig() {
  static MemoryPoolConfig config = [] {
    MemoryPoolConfig base = DefaultPinnedPoolConfig();
    base = LoadMemoryPoolConfig("LATTICE_PINNED_POOL", base);
    base = LoadMemoryPoolConfig("LATTICE_METAL_PINNED_POOL", base);
    return base;
  }();
  return config;
}

}  // namespace

struct MetalBackend::DeviceContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  MetalDeviceDesc desc;
  DeviceCapabilities caps;
  std::string fingerprint;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache;
  std::unique_ptr<MemoryPool> device_pool;
  std::unique_ptr<MemoryPool> pinned_pool;
};

MetalBackend::MetalBackend() = default;

MetalBackend::~MetalBackend() {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto& dev : devices_) {
    if (dev.device_pool) {
      if (dev.device_pool->Outstanding() > 0) {
        LogBackend({LogLevel::kWarn, BackendType::kMetal, BackendErrorKind::kMemory,
                    "device pool has outstanding allocations", "device_pool_leak", dev.desc.index,
                    dev.desc.name});
      }
      dev.device_pool->Trim();
    }
    if (dev.pinned_pool) {
      if (dev.pinned_pool->Outstanding() > 0) {
        LogBackend({LogLevel::kWarn, BackendType::kMetal, BackendErrorKind::kMemory,
                    "pinned pool has outstanding allocations", "pinned_pool_leak", dev.desc.index,
                    dev.desc.name});
      }
      dev.pinned_pool->Trim();
    }
  }
  devices_.clear();
}

BackendType MetalBackend::Type() const {
  return BackendType::kMetal;
}

std::string MetalBackend::Name() const {
  return "Metal";
}

BackendCapabilities MetalBackend::Capabilities() const {
  BackendCapabilities caps;
  caps.supports_dense = true;
  caps.supports_sparse = true;
  caps.supports_ragged = true;
  caps.supports_fft = true;
  caps.supports_blas = true;
  caps.supports_conv = true;
  caps.supports_rng = true;
  caps.supports_events = true;
  caps.supported_dtypes = {DType::kF32, DType::kF64, DType::kI32, DType::kU32};
  return caps;
}

StatusOr<std::shared_ptr<Stream>> MetalBackend::CreateStream() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) {
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                       "No Metal devices available");
  }
  return std::make_shared<MetalStream>();
}

StatusOr<std::shared_ptr<Event>> MetalBackend::CreateEvent() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  return std::make_shared<MetalEvent>();
}

StatusOr<Allocation> MetalBackend::Allocate(size_t bytes, size_t alignment) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) {
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                       "No Metal devices available");
  }
  if (bytes == 0) {
    Allocation empty;
    empty.kind = AllocationKind::kDevice;
    return empty;
  }
  auto* pool = DevicePool(0);
  if (!pool) {
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
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

Status MetalBackend::Deallocate(const Allocation& alloc) const {
  if (!alloc.device_handle) return Status::OK();
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  auto* pool = DevicePool(0);
  if (!pool) {
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                       "Device pool unavailable");
  }
  return pool->Release(reinterpret_cast<uintptr_t>(alloc.device_handle));
}

StatusOr<Allocation> MetalBackend::AllocatePinned(size_t bytes, size_t alignment) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) {
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                       "No Metal devices available");
  }
  if (bytes == 0) {
    Allocation empty;
    empty.kind = AllocationKind::kPinnedHost;
    return empty;
  }
  auto* pool = PinnedPool(0);
  if (!pool) {
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
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

Status MetalBackend::DeallocatePinned(const Allocation& alloc) const {
  if (!alloc.ptr) return Status::OK();
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  auto* pool = PinnedPool(0);
  if (!pool) {
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                       "Pinned pool unavailable");
  }
  return pool->Release(reinterpret_cast<uintptr_t>(alloc.ptr));
}

int MetalBackend::NumThreads() const {
  return 1;
}

size_t MetalBackend::OutstandingAllocs() const {
  size_t total = 0;
  for (int i = 0; i < static_cast<int>(devices_.size()); ++i) {
    auto& dev = devices_[i];
    if (dev.device_pool) total += dev.device_pool->Outstanding();
    if (dev.pinned_pool) total += dev.pinned_pool->Outstanding();
  }
  return total;
}

BackendMemoryStats MetalBackend::MemoryStats() const {
  BackendMemoryStats stats;
  Status status = EnsureInitialized();
  if (!status.ok()) return stats;
  for (int i = 0; i < static_cast<int>(devices_.size()); ++i) {
    auto& dev = devices_[i];
    if (dev.device_pool) {
      AccumulateMemoryPoolStats(&stats.device, dev.device_pool->Stats());
    }
    if (dev.pinned_pool) {
      AccumulateMemoryPoolStats(&stats.pinned, dev.pinned_pool->Stats());
    }
  }
  return stats;
}

void MetalBackend::SetDefaultPriority(int priority) {
  default_priority_ = priority;
}

void MetalBackend::SetDeterministic(bool deterministic) {
  deterministic_ = deterministic;
}

int MetalBackend::DeviceCount() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return 0;
  return static_cast<int>(devices_.size());
}

std::vector<MetalDeviceDesc> MetalBackend::DeviceInfo() const {
  Status status = EnsureInitialized();
  std::vector<MetalDeviceDesc> out;
  if (!status.ok()) return out;
  for (const auto& dev : devices_) {
    out.push_back(dev.desc);
  }
  return out;
}

std::vector<DeviceCapabilities> MetalBackend::DeviceCaps() const {
  Status status = EnsureInitialized();
  std::vector<DeviceCapabilities> out;
  if (!status.ok()) return out;
  for (const auto& dev : devices_) {
    out.push_back(dev.caps);
  }
  return out;
}

StatusOr<MetalBuffer> MetalBackend::CreateBuffer(int device_index, size_t bytes) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return MetalStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                       "Invalid Metal device index");
  }
  if (bytes == 0) {
    return MetalBuffer{nullptr, 0, device_index};
  }
  auto* pool = DevicePool(device_index);
  if (!pool) {
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                       "Device pool unavailable");
  }
  auto block_or = pool->Acquire(bytes, 64);
  if (!block_or.ok()) return block_or.status();
  auto block = block_or.value();
  return MetalBuffer{reinterpret_cast<void*>(block.handle), bytes, device_index};
}

Status MetalBackend::ReleaseBuffer(MetalBuffer* buffer) const {
  if (!buffer || !buffer->handle) return Status::OK();
  auto* pool = DevicePool(buffer->device_index);
  if (!pool) {
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                       "Device pool unavailable");
  }
  Status release = pool->Release(reinterpret_cast<uintptr_t>(buffer->handle));
  if (!release.ok()) return release;
  buffer->handle = nullptr;
  buffer->bytes = 0;
  buffer->device_index = -1;
  return Status::OK();
}

Status MetalBackend::WriteBuffer(int device_index, const MetalBuffer& buffer, const void* data,
                                 size_t bytes, size_t offset) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return MetalStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                       "Invalid Metal device index");
  }
  if (bytes + offset > buffer.bytes) {
    return MetalStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                       "Write exceeds buffer size");
  }
  id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer.handle;
  void* dst = static_cast<char*>(mtl_buffer.contents) + static_cast<std::ptrdiff_t>(offset);
  std::memcpy(dst, data, bytes);
  return Status::OK();
}

Status MetalBackend::ReadBuffer(int device_index, const MetalBuffer& buffer, void* data,
                                size_t bytes, size_t offset) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
    return MetalStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                       "Invalid Metal device index");
  }
  if (bytes + offset > buffer.bytes) {
    return MetalStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                       "Read exceeds buffer size");
  }
  id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer.handle;
  void* src = static_cast<char*>(mtl_buffer.contents) + static_cast<std::ptrdiff_t>(offset);
  std::memcpy(data, src, bytes);
  return Status::OK();
}

StatusOr<MetalKernel> MetalBackend::BuildKernelFromFile(
    const std::string& path, const std::string& kernel_name,
    const std::string& extra_build_options) const {
  auto kernels_or = BuildKernelsFromFile(path, kernel_name, extra_build_options);
  if (!kernels_or.ok()) return kernels_or.status();
  auto kernels = kernels_or.value();
  if (kernels.empty()) {
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kBuild,
                       "No Metal kernels built");
  }
  return kernels.front();
}

StatusOr<std::vector<MetalKernel>> MetalBackend::BuildKernelsFromFile(
    const std::string& path, const std::string& kernel_name,
    const std::string& extra_build_options) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (devices_.empty()) {
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                       "No Metal devices available");
  }

  std::filesystem::path source_path(path);
  std::string source;
  std::string error;
  if (!ReadFile(source_path, &source, &error)) {
    return MetalStatus(StatusCode::kInvalidArgument, BackendErrorKind::kIo, error);
  }

  std::vector<MetalKernel> out;
  std::string last_error;
  for (size_t i = 0; i < devices_.size(); ++i) {
    auto& dev = devices_[i];
    std::string build_options = extra_build_options;
    std::string cache_key = CacheKey(dev, kernel_name, build_options, source);
    auto cache_it = dev.pipeline_cache.find(cache_key);
    id<MTLComputePipelineState> pipeline = nil;
    if (cache_it != dev.pipeline_cache.end()) {
      pipeline = cache_it->second;
    } else {
      KernelTrace trace;
      trace.backend = BackendType::kMetal;
      trace.kernel_name = kernel_name;
      trace.build_options = build_options;
      trace.source = source;
      trace.device_index = dev.desc.index;
      trace.device_name = dev.desc.name;
      std::string trace_path;
      if (TraceKernelSource(trace, &trace_path)) {
        LogBackend({LogLevel::kTrace, BackendType::kMetal, BackendErrorKind::kBuild,
                    "kernel trace written", "build", dev.desc.index, dev.desc.name, 0, "",
                    trace_path});
      }

      MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
      const uint32_t device_index = static_cast<uint32_t>(dev.desc.index);
      NSMutableDictionary<NSString*, NSObject*>* macros = [@{
        @"LATTICE_ABI_VERSION" : [NSString stringWithFormat:@"%u", metal::kAbiVersion],
        @"LATTICE_ABI_VERSION_MIN" : [NSString stringWithFormat:@"%u", metal::kAbiVersionMin],
        @"LATTICE_DEVICE_INDEX" : [NSString stringWithFormat:@"%u", device_index],
      } mutableCopy];
      if (dev.caps.fp16 == CapabilityStatus::kYes) {
        macros[@"LATTICE_HAS_FP16"] = @"1";
      }
      if (dev.caps.fp64 == CapabilityStatus::kYes) {
        macros[@"LATTICE_HAS_FP64"] = @"1";
      }
      options.preprocessorMacros = macros;
      NSError* ns_error = nil;
      id<MTLLibrary> library =
          [dev.device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                   options:options
                                     error:&ns_error];
      if (!library) {
        last_error = ns_error ? ns_error.localizedDescription.UTF8String
                              : "Metal library compilation failed";
        LogBackend({LogLevel::kWarn, BackendType::kMetal, BackendErrorKind::kBuild, last_error,
                    "build", dev.desc.index, dev.desc.name});
        continue;
      }
      id<MTLFunction> function =
          [library newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
      if (!function) {
        last_error = "Metal function not found";
        LogBackend({LogLevel::kWarn, BackendType::kMetal, BackendErrorKind::kBuild, last_error,
                    "build", dev.desc.index, dev.desc.name});
        continue;
      }
      pipeline = [dev.device newComputePipelineStateWithFunction:function error:&ns_error];
      if (!pipeline) {
        last_error =
            ns_error ? ns_error.localizedDescription.UTF8String : "Metal pipeline creation failed";
        LogBackend({LogLevel::kWarn, BackendType::kMetal, BackendErrorKind::kBuild, last_error,
                    "build", dev.desc.index, dev.desc.name});
        continue;
      }
      dev.pipeline_cache.emplace(cache_key, pipeline);
    }
    MetalKernel handle;
    handle.pipeline = (__bridge void*)pipeline;
    handle.device_index = static_cast<int>(i);
    handle.name = kernel_name;
    out.push_back(handle);
  }

  if (out.empty()) {
    if (last_error.empty()) last_error = "No Metal kernels built";
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kBuild, last_error);
  }
  return out;
}

Status MetalBackend::ReleaseKernel(MetalKernel* kernel) const {
  if (!kernel) return Status::OK();
  kernel->pipeline = nullptr;
  return Status::OK();
}

Status MetalBackend::LaunchKernel(const MetalKernel& kernel, const MetalLaunchConfig& config,
                                  const std::vector<MetalKernelArg>& args) const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;
  if (kernel.device_index < 0 || kernel.device_index >= static_cast<int>(devices_.size())) {
    return MetalStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                       "Invalid Metal device index");
  }
  auto& dev = devices_[kernel.device_index];
  id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)kernel.pipeline;
  if (!pipeline) {
    return MetalStatus(StatusCode::kInvalidArgument, BackendErrorKind::kInvalidArgument,
                       "Invalid Metal pipeline");
  }

  id<MTLCommandBuffer> cmd = [dev.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:pipeline];

  for (size_t i = 0; i < args.size(); ++i) {
    const auto& arg = args[i];
    if (arg.kind == MetalKernelArg::Kind::kBuffer) {
      id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)arg.buffer;
      [enc setBuffer:buffer offset:0 atIndex:i];
    } else {
      [enc setBytes:arg.value length:arg.size atIndex:i];
    }
  }

  MTLSize grid = MTLSizeMake(config.grid[0], config.grid[1], config.grid[2]);
  MTLSize threads = MTLSizeMake(config.threads[0], config.threads[1], config.threads[2]);
  if (config.use_threads) {
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
  } else {
    NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger width = std::min(static_cast<NSUInteger>(config.grid[0]), max_threads);
    MTLSize threads_per_group = MTLSizeMake(width, 1, 1);
    NSUInteger groups = (config.grid[0] + width - 1) / width;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:threads_per_group];
  }

  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.error) {
    return MetalStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                       cmd.error.localizedDescription.UTF8String);
  }
  return Status::OK();
}

Status MetalBackend::SmokeTest() const {
  Status status = EnsureInitialized();
  if (!status.ok()) return status;

  std::string kernel_dir = KernelDir();
  if (kernel_dir.empty()) {
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kIo,
                       "Metal kernel directory not found");
  }

  const std::string kernel_path =
      (std::filesystem::path(kernel_dir) / "lattice_smoke.metal").string();
  auto kernels_or = BuildKernelsFromFile(kernel_path, "vec_add");
  if (!kernels_or.ok()) return kernels_or.status();
  auto kernels = kernels_or.value();
  if (kernels.empty()) {
    return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                       "No Metal devices available");
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

    MetalLaunchConfig cfg;
    cfg.grid[0] = static_cast<uint32_t>(kCount);
    std::vector<MetalKernelArg> args;
    args.push_back(MetalKernelArg::Buffer(buf_a.handle));
    args.push_back(MetalKernelArg::Buffer(buf_b.handle));
    args.push_back(MetalKernelArg::Buffer(buf_out.handle));
    unsigned int count = static_cast<unsigned int>(kCount);
    args.push_back(MetalKernelArg::Value(&count, sizeof(count)));

    status = LaunchKernel(kernel, cfg, args);
    if (!status.ok()) return status;

    status = ReadBuffer(kernel.device_index, buf_out, out.data(), out.size() * sizeof(float));
    if (!status.ok()) return status;

    for (size_t i = 0; i < kCount; ++i) {
      if (out[i] != a[i] + b[i]) {
        return MetalStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                           "Metal smoke test failed: incorrect output");
      }
    }

    ReleaseBuffer(&buf_a);
    ReleaseBuffer(&buf_b);
    ReleaseBuffer(&buf_out);
  }

  return Status::OK();
}

Status MetalBackend::EnsureInitialized() const {
  std::lock_guard<std::mutex> lock(mu_);
  if (initialized_) return init_status_;
  initialized_ = true;

  devices_.clear();

  @autoreleasepool {
    NSArray<id<MTLDevice>>* metal_devices = MTLCopyAllDevices();
    if (!metal_devices || [metal_devices count] == 0) {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device) {
        metal_devices = @[ device ];
      }
    }

    NSString* os_version = [[NSProcessInfo processInfo] operatingSystemVersionString];
    const std::string runtime_str = os_version ? os_version.UTF8String : "";
    std::vector<id<MTLDevice>> device_list;
    device_list.reserve(static_cast<size_t>([metal_devices count]));
    for (id<MTLDevice> device in metal_devices) {
      device_list.push_back(device);
    }

    std::vector<DeviceIdentity> identities;
    identities.reserve(device_list.size());
    for (size_t i = 0; i < device_list.size(); ++i) {
      id<MTLDevice> device = device_list[i];
      DeviceIdentity identity;
      identity.index = static_cast<int>(i);
      identity.name = device.name.UTF8String;
      identity.vendor = "Apple";
      identity.driver = runtime_str;
      identity.kind = DeviceKind::kGPU;
      identities.push_back(identity);
    }

    DeviceSelectionOptions selection = LoadDeviceSelectionOptions("LATTICE_METAL");
    DeviceSelectionResult selected = SelectDevices(identities, selection);
    if (selected.indices.empty()) {
      init_status_ = MetalStatus(
          StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
          selected.diagnostics.empty() ? "No Metal devices selected" : selected.diagnostics);
      return init_status_;
    }

    for (int idx : selected.indices) {
      if (idx < 0 || idx >= static_cast<int>(device_list.size())) continue;
      id<MTLDevice> device = device_list[static_cast<size_t>(idx)];
      DeviceContext ctx;
      ctx.device = device;
      ctx.desc.index = idx;
      ctx.desc.name = device.name.UTF8String;
      ctx.desc.vendor = "Apple";
      ctx.desc.driver_version = runtime_str;
      ctx.desc.runtime_version = runtime_str;
      ctx.desc.max_threadgroup_size = device.maxThreadsPerThreadgroup.width;
      ctx.desc.shared_mem_bytes = device.maxThreadgroupMemoryLength;
      ctx.caps.is_gpu = true;
      ctx.caps.local_mem_bytes = ctx.desc.shared_mem_bytes;
      ctx.caps.max_work_item_sizes[0] = device.maxThreadsPerThreadgroup.width;
      ctx.caps.max_work_item_sizes[1] = device.maxThreadsPerThreadgroup.height;
      ctx.caps.max_work_item_sizes[2] = device.maxThreadsPerThreadgroup.depth;
      ctx.caps.max_work_group_size = device.maxThreadsPerThreadgroup.width *
                                     device.maxThreadsPerThreadgroup.height *
                                     device.maxThreadsPerThreadgroup.depth;
      ctx.caps.max_threads_per_block = ctx.caps.max_work_group_size;
      bool feature = false;
      if (QueryBoolSelector(device, @selector(supports64BitFloat), &feature)) {
        ctx.caps.fp64 = feature ? CapabilityStatus::kYes : CapabilityStatus::kNo;
      }
      if (QueryBoolSelector(device, @selector(supports16BitFloat), &feature)) {
        ctx.caps.fp16 = feature ? CapabilityStatus::kYes : CapabilityStatus::kNo;
      }
      ctx.caps.quirks =
          QueryDeviceQuirks(BackendType::kMetal, ctx.desc.vendor, ctx.desc.name, runtime_str);
      ctx.caps.is_software = (ctx.caps.quirks.flags & kSoftwareEmulation) != 0;
      if (ctx.caps.quirks.flags & kDisableFp16) ctx.caps.fp16 = CapabilityStatus::kNo;
      if (ctx.caps.quirks.flags & kDisableFp64) ctx.caps.fp64 = CapabilityStatus::kNo;

      DeviceMetadata meta = BuildDeviceMetadata(ctx.desc, ctx.caps);
      ctx.fingerprint = DeviceFingerprint(meta);

      if (ctx.caps.quirks.disabled) {
        LogBackend({LogLevel::kWarn, BackendType::kMetal, BackendErrorKind::kDiscovery,
                    "skipping device: " + ctx.caps.quirks.reason, "device_skip", ctx.desc.index,
                    ctx.desc.name});
        continue;
      }
      ctx.queue = [device newCommandQueue];
      if (!ctx.queue) {
        LogBackend({LogLevel::kWarn, BackendType::kMetal, BackendErrorKind::kContext,
                    "queue init failed", "queue", ctx.desc.index, ctx.desc.name});
        continue;
      }
      devices_.push_back(std::move(ctx));
    }
  }

  if (devices_.empty()) {
    init_status_ = MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kContext,
                               "No Metal devices initialized");
    return init_status_;
  }

  {
    DeviceMetadataStore meta_store;
    std::string meta_error;
    for (const auto& dev : devices_) {
      const DeviceMetadata meta = BuildDeviceMetadata(dev.desc, dev.caps);
      if (!meta_store.Write(meta, &meta_error)) {
        LogBackend({LogLevel::kWarn, BackendType::kMetal, BackendErrorKind::kIo,
                    "metadata persist failed: " + meta_error, "metadata", dev.desc.index,
                    dev.desc.name});
        meta_error.clear();
      }
    }
  }

  for (size_t i = 0; i < devices_.size(); ++i) {
    const auto& dev = devices_[i];
    LogBackend({LogLevel::kInfo, BackendType::kMetal, BackendErrorKind::kDiscovery, dev.desc.name,
                "device_info", dev.desc.index, dev.desc.name});
  }

  init_status_ = Status::OK();
  return init_status_;
}

MemoryPool* MetalBackend::DevicePool(int device_index) const {
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) return nullptr;
  std::lock_guard<std::mutex> lock(mu_);
  auto& dev = devices_[device_index];
  if (dev.device_pool) return dev.device_pool.get();

  MemoryPoolConfig config = MetalDevicePoolConfig();
  const int idx = device_index;
  auto alloc_fn = [this, idx](size_t bytes, size_t alignment) -> StatusOr<PoolBlock> {
    (void)alignment;
    auto& device = devices_[idx];
    id<MTLBuffer> buffer = [device.device newBufferWithLength:bytes
                                                      options:MTLResourceStorageModeShared];
    if (!buffer) {
      return MetalStatus(StatusCode::kInternal, BackendErrorKind::kMemory,
                         "Failed to allocate Metal buffer");
    }
    void* handle = (__bridge_retained void*)buffer;
    PoolBlock block;
    block.key = reinterpret_cast<uintptr_t>(handle);
    block.handle = reinterpret_cast<uintptr_t>(handle);
    block.bytes = bytes;
    block.alignment = alignment;
    return block;
  };
  auto free_fn = [](const PoolBlock& block) -> Status {
    if (!block.handle) return Status::OK();
    CFRelease(reinterpret_cast<void*>(block.handle));
    return Status::OK();
  };
  auto scrub_fn = [this, idx](const PoolBlock& block) -> Status {
    if (!block.handle || block.bytes == 0) return Status::OK();
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(block.handle);
    void* dst = buffer.contents;
    if (dst) {
      std::memset(dst, 0, block.bytes);
    }
    return Status::OK();
  };

  std::ostringstream label;
  label << "metal_device_pool_" << dev.desc.index;
  dev.device_pool = std::make_unique<MemoryPool>(label.str(), config, alloc_fn, free_fn, scrub_fn);
  return dev.device_pool.get();
}

MemoryPool* MetalBackend::PinnedPool(int device_index) const {
  if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) return nullptr;
  std::lock_guard<std::mutex> lock(mu_);
  auto& dev = devices_[device_index];
  if (dev.pinned_pool) return dev.pinned_pool.get();

  MemoryPoolConfig config = MetalPinnedPoolConfig();
  const int idx = device_index;
  auto alloc_fn = [this, idx](size_t bytes, size_t alignment) -> StatusOr<PoolBlock> {
    (void)alignment;
    auto& device = devices_[idx];
    MTLResourceOptions options =
        MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
    id<MTLBuffer> buffer = [device.device newBufferWithLength:bytes options:options];
    if (!buffer) {
      return MetalStatus(StatusCode::kInternal, BackendErrorKind::kMemory,
                         "Failed to allocate Metal pinned buffer");
    }
    void* host_ptr = buffer.contents;
    if (!host_ptr) {
      return MetalStatus(StatusCode::kInternal, BackendErrorKind::kMemory,
                         "Metal pinned buffer has no host pointer");
    }
    void* handle = (__bridge_retained void*)buffer;
    PoolBlock block;
    block.key = reinterpret_cast<uintptr_t>(host_ptr);
    block.handle = reinterpret_cast<uintptr_t>(handle);
    block.host_ptr = host_ptr;
    block.bytes = bytes;
    block.alignment = alignment;
    return block;
  };
  auto free_fn = [](const PoolBlock& block) -> Status {
    if (!block.handle) return Status::OK();
    CFRelease(reinterpret_cast<void*>(block.handle));
    return Status::OK();
  };
  auto scrub_fn = [](const PoolBlock& block) -> Status {
    if (block.host_ptr && block.bytes > 0) {
      std::memset(block.host_ptr, 0, block.bytes);
    }
    return Status::OK();
  };

  std::ostringstream label;
  label << "metal_pinned_pool_" << dev.desc.index;
  dev.pinned_pool = std::make_unique<MemoryPool>(label.str(), config, alloc_fn, free_fn, scrub_fn);
  return dev.pinned_pool.get();
}

std::string MetalBackend::KernelDir() const {
  if (const char* env = std::getenv("LATTICE_KERNEL_DIR")) {
    return std::string(env);
  }
  std::filesystem::path cwd = std::filesystem::current_path();
  for (int i = 0; i < 4; ++i) {
    std::filesystem::path candidate = cwd / "Metal";
    if (std::filesystem::exists(candidate)) {
      return candidate.string();
    }
    cwd = cwd.parent_path();
  }
  return "";
}

std::string MetalBackend::CacheKey(const DeviceContext& dev, const std::string& kernel_name,
                                   const std::string& build_options,
                                   const std::string& source) const {
  std::ostringstream meta;
  meta << dev.desc.name << "|" << kernel_name << "|" << build_options;
  uint64_t meta_hash = Fnv1a64(meta.str());
  uint64_t src_hash = Fnv1a64(source);
  return "metal_" + Hex64(meta_hash) + "_" + Hex64(src_hash);
}

const Backend* GetMetalBackend() {
  static MetalBackend* backend = [] { return new MetalBackend(); }();
  return backend;
}

Status RunMetalSmokeTest() {
  const auto* backend = static_cast<const MetalBackend*>(GetMetalBackend());
  return backend->SmokeTest();
}

}  // namespace lattice::runtime
