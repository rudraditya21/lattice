#include "runtime/backends/metal_backend.h"

#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <objc/message.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "runtime/backends/backend_error.h"
#include "runtime/backends/backend_log.h"
#include "runtime/backends/cache_store.h"
#include "runtime/backends/device_quirks.h"
#include "runtime/backends/device_selector.h"
#include "runtime/backends/execution_config.h"
#include "runtime/backends/kernel_build.h"
#include "runtime/backends/memory_pool.h"
#include "runtime/backends/memory_stats.h"
#include "runtime/backends/memory_utils.h"
#include "runtime/backends/metal_abi.h"
#include "runtime/backends/profiling.h"

namespace lattice::runtime {

namespace {

Status MetalStatus(StatusCode code,
                   BackendErrorKind kind,
                   const std::string& message) {
    return MakeBackendError(code, BackendType::kMetal, kind, message);
}

Status MetalErrorStatus(StatusCode code,
                        BackendErrorKind kind,
                        const std::string& message,
                        NSError* error) {
    std::string detail = message;
    if (error && error.localizedDescription) {
        detail += ": ";
        detail += error.localizedDescription.UTF8String;
    }
    Status st = MetalStatus(code, kind, detail);
    if (error) {
        st.backend_code = static_cast<int64_t>(error.code);
        if (error.domain) {
            st.backend_error_name = error.domain.UTF8String;
        }
    }
    return st;
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

bool ReadFile(const std::filesystem::path& path,
              std::string* out,
              std::string* error) {
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

std::string MetalDeviceInfoJson(const MetalDeviceDesc& desc,
                                const DeviceCapabilities& caps) {
    std::ostringstream out;
    out << "{";
    out << "\"name\":\"" << EscapeJson(desc.name) << "\"";
    if (!desc.vendor.empty()) {
        out << ",\"vendor\":\"" << EscapeJson(desc.vendor) << "\"";
    }
    if (!desc.driver_version.empty()) {
        out << ",\"driver_version\":\"" << EscapeJson(desc.driver_version)
            << "\"";
    }
    if (!desc.runtime_version.empty()) {
        out << ",\"runtime_version\":\"" << EscapeJson(desc.runtime_version)
            << "\"";
    }
    out << ",\"max_threadgroup_size\":" << desc.max_threadgroup_size;
    out << ",\"shared_mem_bytes\":" << desc.shared_mem_bytes;
    out << ",\"fp16\":" << CapabilityToInt(caps.fp16);
    out << ",\"fp64\":" << CapabilityToInt(caps.fp64);
    out << ",\"local_mem_bytes\":" << caps.local_mem_bytes;
    out << ",\"max_work_group_size\":" << caps.max_work_group_size;
    out << ",\"max_work_item_sizes\":[" << caps.max_work_item_sizes[0] << ","
        << caps.max_work_item_sizes[1] << "," << caps.max_work_item_sizes[2]
        << "]";
    out << ",\"is_gpu\":" << (caps.is_gpu ? "true" : "false");
    out << ",\"is_cpu\":" << (caps.is_cpu ? "true" : "false");
    out << ",\"is_software\":" << (caps.is_software ? "true" : "false");
    out << ",\"quirks_flags\":" << caps.quirks.flags;
    out << ",\"quirks_disabled\":" << (caps.quirks.disabled ? "true" : "false");
    if (!caps.quirks.reason.empty()) {
        out << ",\"quirks_reason\":\"" << EscapeJson(caps.quirks.reason)
            << "\"";
    }
    out << "}";
    return out.str();
}

std::unordered_map<std::string, std::string> ParseDefineOptions(
    const std::string& options) {
    std::unordered_map<std::string, std::string> out;
    std::istringstream in(options);
    std::string token;
    while (in >> token) {
        if (token.rfind("-D", 0) != 0)
            continue;
        std::string def = token.substr(2);
        if (def.empty())
            continue;
        const size_t eq = def.find('=');
        if (eq == std::string::npos) {
            out[def] = "1";
        } else {
            out[def.substr(0, eq)] = def.substr(eq + 1);
        }
    }
    return out;
}

class MetalEvent final : public Event {
   public:
    MetalEvent(id<MTLCommandQueue> queue,
               ProfilingState* profiling,
               int device_index)
        : queue_(queue), profiling_(profiling), device_index_(device_index) {}

    void Record() override {
        ProfilingScope scope(profiling_, {ProfilingEventKind::kEventRecord,
                                          BackendType::kMetal, device_index_});
        ready_.store(false, std::memory_order_relaxed);
        if (!queue_) {
            scope.SetStatus(StatusCode::kUnavailable);
            ready_.store(true, std::memory_order_relaxed);
            return;
        }
        cmd_ = [queue_ commandBuffer];
        if (!cmd_) {
            scope.SetStatus(StatusCode::kInternal);
            ready_.store(true, std::memory_order_relaxed);
            return;
        }
        std::atomic<bool>* ready_ptr = &ready_;
        [cmd_ addCompletedHandler:^(id<MTLCommandBuffer>) {
          ready_ptr->store(true, std::memory_order_relaxed);
        }];
        [cmd_ commit];
    }

    void Wait() override {
        ProfilingScope scope(profiling_, {ProfilingEventKind::kEventWait,
                                          BackendType::kMetal, device_index_});
        if (cmd_) {
            [cmd_ waitUntilCompleted];
        }
        ready_.store(true, std::memory_order_relaxed);
    }

    bool Ready() const override {
        if (cmd_) {
            if (ready_.load(std::memory_order_relaxed))
                return true;
            return [cmd_ status] == MTLCommandBufferStatusCompleted;
        }
        return ready_.load(std::memory_order_relaxed);
    }

    id<MTLCommandBuffer> handle() const { return cmd_; }

   private:
    id<MTLCommandQueue> queue_ = nil;
    id<MTLCommandBuffer> cmd_ = nil;
    std::atomic<bool> ready_{false};
    ProfilingState* profiling_ = nullptr;
    int device_index_ = -1;
};

class MetalStream final : public Stream {
   public:
    MetalStream(id<MTLCommandQueue> queue,
                ProfilingState* profiling,
                int device_index)
        : queue_(queue), profiling_(profiling), device_index_(device_index) {}
    void Submit(std::function<void()> fn) override {
        for (auto& dep : deps_) {
            dep->Wait();
        }
        deps_.clear();
        fn();
    }
    void Synchronize() override {
        ProfilingScope scope(profiling_, {ProfilingEventKind::kStreamSync,
                                          BackendType::kMetal, device_index_});
        if (queue_) {
            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        deps_.clear();
    }
    void AddDependency(const std::shared_ptr<Event>& ev) override {
        if (!ev)
            return;
        deps_.push_back(ev);
    }
    StatusOr<std::shared_ptr<Event>> CreateEvent() const override {
        if (!queue_) {
            return Status::Unavailable("Metal queue unavailable");
        }
        return std::make_shared<MetalEvent>(queue_, profiling_, device_index_);
    }
    void RecordEvent(const std::shared_ptr<Event>& ev) override {
        if (!ev)
            return;
        Submit([ev]() { ev->Record(); });
    }
    void SetPriority(int priority) override { priority_ = priority; }

   private:
    id<MTLCommandQueue> queue_ = nil;
    int priority_ = 0;
    std::vector<std::shared_ptr<Event>> deps_;
    ProfilingState* profiling_ = nullptr;
    int device_index_ = -1;
};

bool QueryBoolSelector(id obj, SEL sel, bool* out) {
    if (![obj respondsToSelector:sel])
        return false;
    BOOL value = ((BOOL (*)(id, SEL))objc_msgSend)(obj, sel);
    *out = value != NO;
    return true;
}

DeviceMetadata BuildDeviceMetadata(const MetalDeviceDesc& desc,
                                   const DeviceCapabilities& caps) {
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

MetalBackend::MetalBackend() {
    exec_config_ = LoadExecutionConfig("LATTICE", exec_config_);
    exec_config_ = LoadExecutionConfig("LATTICE_METAL", exec_config_);
    profiling_ = std::make_shared<ProfilingState>(BackendType::kMetal);
    profiling_->SetEnabled(exec_config_.enable_profiling);
}

MetalBackend::~MetalBackend() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& dev : devices_) {
        if (dev.device_pool) {
            MemoryPoolStats stats = dev.device_pool->Stats();
            if (HasOutstandingAllocs(stats)) {
                LogBackend({LogLevel::kWarn, BackendType::kMetal,
                            BackendErrorKind::kMemory,
                            "device pool has outstanding allocations (" +
                                FormatPoolStats(stats) + ")",
                            "device_pool_leak", dev.desc.index, dev.desc.name});
            }
            dev.device_pool->Trim();
        }
        if (dev.pinned_pool) {
            MemoryPoolStats stats = dev.pinned_pool->Stats();
            if (HasOutstandingAllocs(stats)) {
                LogBackend({LogLevel::kWarn, BackendType::kMetal,
                            BackendErrorKind::kMemory,
                            "pinned pool has outstanding allocations (" +
                                FormatPoolStats(stats) + ")",
                            "pinned_pool_leak", dev.desc.index, dev.desc.name});
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
    caps.supports_profiling = true;
    caps.supported_dtypes = {DType::kF32, DType::kF64, DType::kI32,
                             DType::kU32};
    return caps;
}

StatusOr<std::shared_ptr<Stream>> MetalBackend::CreateStream() const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    if (devices_.empty()) {
        return MetalStatus(StatusCode::kUnavailable,
                           BackendErrorKind::kDiscovery,
                           "No Metal devices available");
    }
    const auto& dev = devices_[0];
    id<MTLCommandQueue> queue = nil;
    if (dev.device) {
        queue = [dev.device newCommandQueue];
    }
    if (!queue) {
        queue = dev.queue;
    }
    if (!queue) {
        return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kContext,
                           "Metal queue unavailable");
    }
    return std::make_shared<MetalStream>(queue, profiling_.get(),
                                         dev.desc.index);
}

StatusOr<std::shared_ptr<Event>> MetalBackend::CreateEvent() const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    if (devices_.empty()) {
        return MetalStatus(StatusCode::kUnavailable,
                           BackendErrorKind::kDiscovery,
                           "No Metal devices available");
    }
    return std::make_shared<MetalEvent>(devices_[0].queue, profiling_.get(),
                                        devices_[0].desc.index);
}

StatusOr<Allocation> MetalBackend::Allocate(size_t bytes,
                                            size_t alignment) const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    if (devices_.empty()) {
        return MetalStatus(StatusCode::kUnavailable,
                           BackendErrorKind::kDiscovery,
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
    if (!block_or.ok())
        return block_or.status();
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
    if (!alloc.device_handle)
        return Status::OK();
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    auto* pool = DevicePool(0);
    if (!pool) {
        return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                           "Device pool unavailable");
    }
    return pool->Release(reinterpret_cast<uintptr_t>(alloc.device_handle));
}

StatusOr<Allocation> MetalBackend::AllocatePinned(size_t bytes,
                                                  size_t alignment) const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    if (devices_.empty()) {
        return MetalStatus(StatusCode::kUnavailable,
                           BackendErrorKind::kDiscovery,
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
    if (!block_or.ok())
        return block_or.status();
    auto block = block_or.value();
    Allocation alloc;
    alloc.ptr = block.host_ptr;
    alloc.device_handle = reinterpret_cast<void*>(
        block.device_ptr ? block.device_ptr : block.handle);
    alloc.bytes = bytes;
    alloc.alignment = alignment;
    alloc.from_pool = block.from_pool;
    alloc.kind = AllocationKind::kPinnedHost;
    return alloc;
}

Status MetalBackend::DeallocatePinned(const Allocation& alloc) const {
    if (!alloc.ptr)
        return Status::OK();
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
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
        if (dev.device_pool)
            total += dev.device_pool->Outstanding();
        if (dev.pinned_pool)
            total += dev.pinned_pool->Outstanding();
    }
    return total;
}

BackendMemoryStats MetalBackend::MemoryStats() const {
    BackendMemoryStats stats;
    Status status = EnsureInitialized();
    if (!status.ok())
        return stats;
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

std::vector<DeviceMemoryStats> MetalBackend::MemoryStatsByDevice() const {
    std::vector<DeviceMemoryStats> out;
    Status status = EnsureInitialized();
    if (!status.ok())
        return out;
    out.reserve(devices_.size());
    for (const auto& dev : devices_) {
        DeviceMemoryStats entry;
        entry.backend = BackendType::kMetal;
        entry.device_index = dev.desc.index;
        entry.device_name = dev.desc.name;
        if (dev.device_pool) {
            entry.device = dev.device_pool->Stats();
        }
        if (dev.pinned_pool) {
            entry.pinned = dev.pinned_pool->Stats();
        }
        out.push_back(std::move(entry));
    }
    return out;
}

ExecutionConfig MetalBackend::GetExecutionConfig() const {
    std::lock_guard<std::mutex> lock(config_mu_);
    return exec_config_;
}

Status MetalBackend::SetExecutionConfig(const ExecutionConfig& config) {
    {
        std::lock_guard<std::mutex> lock(config_mu_);
        exec_config_ = config;
    }
    if (profiling_) {
        profiling_->SetEnabled(exec_config_.enable_profiling);
    }
    return Status::OK();
}

ProfilingCounters MetalBackend::ProfilingStats() const {
    if (!profiling_) {
        return ProfilingCounters{};
    }
    return profiling_->Snapshot();
}

void MetalBackend::ResetProfilingStats() {
    if (profiling_) {
        profiling_->Reset();
    }
}

void MetalBackend::SetProfilingHook(ProfilingHook hook) {
    if (profiling_) {
        profiling_->SetHook(std::move(hook));
    }
}

StatusOr<uint64_t> MetalBackend::ElapsedNs(
    const std::shared_ptr<Event>& start,
    const std::shared_ptr<Event>& end) const {
    if (!start || !end) {
        return Status::Invalid("Missing events for elapsed time");
    }
    ExecutionConfig config = GetExecutionConfig();
    if (!config.enable_profiling) {
        return Status::Unavailable("Metal profiling disabled");
    }
    auto* start_ev = dynamic_cast<MetalEvent*>(start.get());
    auto* end_ev = dynamic_cast<MetalEvent*>(end.get());
    if (!start_ev || !end_ev) {
        return Status::Invalid("Events do not belong to Metal backend");
    }
    id<MTLCommandBuffer> start_cmd = start_ev->handle();
    id<MTLCommandBuffer> end_cmd = end_ev->handle();
    if (!start_cmd || !end_cmd) {
        return Status::Invalid("Events have no Metal command buffer");
    }
    if ([end_cmd status] != MTLCommandBufferStatusCompleted) {
        [end_cmd waitUntilCompleted];
    }
    if (![start_cmd respondsToSelector:@selector(GPUStartTime)] ||
        ![end_cmd respondsToSelector:@selector(GPUEndTime)]) {
        return Status::Unavailable("Metal GPU timing not available");
    }
    const double start_time = start_cmd.GPUStartTime;
    const double end_time = end_cmd.GPUEndTime;
    if (end_time < start_time) {
        return Status::Invalid("Metal event timestamps invalid");
    }
    double elapsed_sec = end_time - start_time;
    return static_cast<uint64_t>(elapsed_sec * 1e9);
}

void MetalBackend::SetDefaultPriority(int priority) {
    default_priority_ = priority;
}

void MetalBackend::SetDeterministic(bool deterministic) {
    deterministic_ = deterministic;
}

int MetalBackend::DeviceCount() const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return 0;
    return static_cast<int>(devices_.size());
}

std::vector<MetalDeviceDesc> MetalBackend::DeviceInfo() const {
    Status status = EnsureInitialized();
    std::vector<MetalDeviceDesc> out;
    if (!status.ok())
        return out;
    for (const auto& dev : devices_) {
        out.push_back(dev.desc);
    }
    return out;
}

std::vector<DeviceCapabilities> MetalBackend::DeviceCaps() const {
    Status status = EnsureInitialized();
    std::vector<DeviceCapabilities> out;
    if (!status.ok())
        return out;
    for (const auto& dev : devices_) {
        out.push_back(dev.caps);
    }
    return out;
}

StatusOr<MetalBuffer> MetalBackend::CreateBuffer(int device_index,
                                                 size_t bytes) const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
        return MetalStatus(StatusCode::kInvalidArgument,
                           BackendErrorKind::kInvalidArgument,
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
    if (!block_or.ok())
        return block_or.status();
    auto block = block_or.value();
    return MetalBuffer{reinterpret_cast<void*>(block.handle), bytes,
                       device_index};
}

Status MetalBackend::ReleaseBuffer(MetalBuffer* buffer) const {
    if (!buffer || !buffer->handle)
        return Status::OK();
    auto* pool = DevicePool(buffer->device_index);
    if (!pool) {
        return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                           "Device pool unavailable");
    }
    Status release = pool->Release(reinterpret_cast<uintptr_t>(buffer->handle));
    if (!release.ok())
        return release;
    buffer->handle = nullptr;
    buffer->bytes = 0;
    buffer->device_index = -1;
    return Status::OK();
}

Status MetalBackend::WriteBuffer(int device_index,
                                 const MetalBuffer& buffer,
                                 const void* data,
                                 size_t bytes,
                                 size_t offset) const {
    ProfilingScope scope(profiling_.get(), {ProfilingEventKind::kMemcpyH2D,
                                            BackendType::kMetal, device_index});
    scope.SetBytes(bytes);
    Status status = EnsureInitialized();
    if (!status.ok()) {
        scope.SetStatus(status.code);
        return status;
    }
    if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
        Status err = MetalStatus(StatusCode::kInvalidArgument,
                                 BackendErrorKind::kInvalidArgument,
                                 "Invalid Metal device index");
        scope.SetStatus(err.code);
        return err;
    }
    if (bytes + offset > buffer.bytes) {
        Status err = MetalStatus(StatusCode::kInvalidArgument,
                                 BackendErrorKind::kInvalidArgument,
                                 "Write exceeds buffer size");
        scope.SetStatus(err.code);
        return err;
    }
    id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer.handle;
    void* dst = static_cast<char*>(mtl_buffer.contents) +
                static_cast<std::ptrdiff_t>(offset);
    std::memcpy(dst, data, bytes);
    return Status::OK();
}

Status MetalBackend::ReadBuffer(int device_index,
                                const MetalBuffer& buffer,
                                void* data,
                                size_t bytes,
                                size_t offset) const {
    ProfilingScope scope(profiling_.get(), {ProfilingEventKind::kMemcpyD2H,
                                            BackendType::kMetal, device_index});
    scope.SetBytes(bytes);
    Status status = EnsureInitialized();
    if (!status.ok()) {
        scope.SetStatus(status.code);
        return status;
    }
    if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
        Status err = MetalStatus(StatusCode::kInvalidArgument,
                                 BackendErrorKind::kInvalidArgument,
                                 "Invalid Metal device index");
        scope.SetStatus(err.code);
        return err;
    }
    if (bytes + offset > buffer.bytes) {
        Status err = MetalStatus(StatusCode::kInvalidArgument,
                                 BackendErrorKind::kInvalidArgument,
                                 "Read exceeds buffer size");
        scope.SetStatus(err.code);
        return err;
    }
    id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer.handle;
    void* src = static_cast<char*>(mtl_buffer.contents) +
                static_cast<std::ptrdiff_t>(offset);
    std::memcpy(data, src, bytes);
    return Status::OK();
}

StatusOr<bool> MetalBackend::BlasMatmul(int device_index,
                                        const MetalBuffer& a,
                                        const MetalBuffer& b,
                                        const MetalBuffer& c,
                                        int64_t m,
                                        int64_t n,
                                        int64_t k,
                                        bool use_fp64) const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
        return MetalStatus(StatusCode::kInvalidArgument,
                           BackendErrorKind::kInvalidArgument,
                           "Invalid Metal device index");
    }
    if (m <= 0 || n <= 0 || k <= 0) {
        return true;
    }
    if (use_fp64) {
        return Status::Unavailable("Metal BLAS does not support FP64");
    }
    if (!a.handle || !b.handle || !c.handle) {
        return MetalStatus(StatusCode::kInvalidArgument,
                           BackendErrorKind::kInvalidArgument,
                           "Metal BLAS buffers are null");
    }
    if (a.device_index != device_index || b.device_index != device_index ||
        c.device_index != device_index) {
        return MetalStatus(StatusCode::kInvalidArgument,
                           BackendErrorKind::kInvalidArgument,
                           "Metal BLAS buffers belong to a different device");
    }

    auto& dev = devices_[device_index];
    if (!dev.device || !dev.queue) {
        return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kContext,
                           "Metal queue unavailable");
    }
    if (!MPSSupportsMTLDevice(dev.device)) {
        return Status::Unavailable("MPS not supported on this device");
    }

    const size_t m_sz = static_cast<size_t>(m);
    const size_t n_sz = static_cast<size_t>(n);
    const size_t k_sz = static_cast<size_t>(k);
    const size_t elem_size = sizeof(float);
    const size_t max_size = std::numeric_limits<size_t>::max();
    if (m_sz > max_size / k_sz || m_sz > max_size / n_sz ||
        k_sz > max_size / n_sz) {
        return MetalStatus(StatusCode::kInvalidArgument,
                           BackendErrorKind::kInvalidArgument,
                           "Metal BLAS dimensions overflow");
    }
    const size_t count_a = m_sz * k_sz;
    const size_t count_b = k_sz * n_sz;
    const size_t count_c = m_sz * n_sz;
    if (count_a > max_size / elem_size || count_b > max_size / elem_size ||
        count_c > max_size / elem_size) {
        return MetalStatus(StatusCode::kInvalidArgument,
                           BackendErrorKind::kInvalidArgument,
                           "Metal BLAS size overflow");
    }
    const size_t bytes_a = count_a * elem_size;
    const size_t bytes_b = count_b * elem_size;
    const size_t bytes_c = count_c * elem_size;
    if (bytes_a > a.bytes || bytes_b > b.bytes || bytes_c > c.bytes) {
        return MetalStatus(StatusCode::kInvalidArgument,
                           BackendErrorKind::kInvalidArgument,
                           "Metal BLAS buffers are too small");
    }
    if (m_sz > std::numeric_limits<NSUInteger>::max() ||
        n_sz > std::numeric_limits<NSUInteger>::max() ||
        k_sz > std::numeric_limits<NSUInteger>::max()) {
        return MetalStatus(StatusCode::kInvalidArgument,
                           BackendErrorKind::kInvalidArgument,
                           "Metal BLAS dimensions exceed NSUInteger");
    }

    const NSUInteger m_u = static_cast<NSUInteger>(m_sz);
    const NSUInteger n_u = static_cast<NSUInteger>(n_sz);
    const NSUInteger k_u = static_cast<NSUInteger>(k_sz);
    const NSUInteger row_bytes_a = static_cast<NSUInteger>(k_sz * elem_size);
    const NSUInteger row_bytes_b = static_cast<NSUInteger>(n_sz * elem_size);
    const NSUInteger row_bytes_c = static_cast<NSUInteger>(n_sz * elem_size);

    id<MTLBuffer> a_buf = (__bridge id<MTLBuffer>)a.handle;
    id<MTLBuffer> b_buf = (__bridge id<MTLBuffer>)b.handle;
    id<MTLBuffer> c_buf = (__bridge id<MTLBuffer>)c.handle;
    if (!a_buf || !b_buf || !c_buf) {
        return MetalStatus(StatusCode::kInvalidArgument,
                           BackendErrorKind::kInvalidArgument,
                           "Metal BLAS buffers are invalid");
    }

    MPSMatrixDescriptor* a_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:m_u
                                              columns:k_u
                                             rowBytes:row_bytes_a
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* b_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:k_u
                                              columns:n_u
                                             rowBytes:row_bytes_b
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* c_desc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:m_u
                                              columns:n_u
                                             rowBytes:row_bytes_c
                                             dataType:MPSDataTypeFloat32];
    if (!a_desc || !b_desc || !c_desc) {
        return MetalStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                           "Failed to create MPS matrix descriptors");
    }

    MPSMatrix* a_mat = [[MPSMatrix alloc] initWithBuffer:a_buf
                                                  offset:0
                                              descriptor:a_desc];
    MPSMatrix* b_mat = [[MPSMatrix alloc] initWithBuffer:b_buf
                                                  offset:0
                                              descriptor:b_desc];
    MPSMatrix* c_mat = [[MPSMatrix alloc] initWithBuffer:c_buf
                                                  offset:0
                                              descriptor:c_desc];
    if (!a_mat || !b_mat || !c_mat) {
        return MetalStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                           "Failed to create MPS matrices");
    }

    MPSMatrixMultiplication* op =
        [[MPSMatrixMultiplication alloc] initWithDevice:dev.device
                                          transposeLeft:false
                                         transposeRight:false
                                             resultRows:m_u
                                          resultColumns:n_u
                                        interiorColumns:k_u
                                                  alpha:1.0f
                                                   beta:0.0f];
    if (!op) {
        return MetalStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                           "Failed to create MPS matmul");
    }

    id<MTLCommandBuffer> cmd = [dev.queue commandBuffer];
    if (!cmd) {
        return MetalStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                           "Failed to create Metal command buffer");
    }
    [op encodeToCommandBuffer:cmd
                   leftMatrix:a_mat
                  rightMatrix:b_mat
                 resultMatrix:c_mat];
    [cmd commit];
    [cmd waitUntilCompleted];

    NSError* error = cmd.error;
    if (error) {
        return MetalErrorStatus(StatusCode::kInternal,
                                BackendErrorKind::kRuntime,
                                "Metal BLAS command buffer failed", error);
    }

    return true;
}

StatusOr<bool> MetalBackend::BlasCopy(int device_index,
                                      const MetalBuffer& src,
                                      const MetalBuffer& dst,
                                      int64_t count,
                                      bool use_fp64) const {
    (void)device_index;
    (void)src;
    (void)dst;
    (void)count;
    (void)use_fp64;
    return Status::Unavailable("Metal BLAS copy unavailable");
}

StatusOr<bool> MetalBackend::BlasAxpy(int device_index,
                                      const MetalBuffer& x,
                                      const MetalBuffer& y,
                                      int64_t count,
                                      double alpha,
                                      bool use_fp64) const {
    (void)device_index;
    (void)x;
    (void)y;
    (void)count;
    (void)alpha;
    (void)use_fp64;
    return Status::Unavailable("Metal BLAS axpy unavailable");
}

StatusOr<MetalKernel> MetalBackend::BuildKernelFromFile(
    const std::string& path,
    const std::string& kernel_name,
    const std::string& extra_build_options) const {
    auto kernels_or =
        BuildKernelsFromFile(path, kernel_name, extra_build_options);
    if (!kernels_or.ok())
        return kernels_or.status();
    auto kernels = kernels_or.value();
    if (kernels.empty()) {
        return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kBuild,
                           "No Metal kernels built");
    }
    return kernels.front();
}

StatusOr<std::vector<MetalKernel>> MetalBackend::BuildKernelsFromFile(
    const std::string& path,
    const std::string& kernel_name,
    const std::string& extra_build_options) const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    if (devices_.empty()) {
        return MetalStatus(StatusCode::kUnavailable,
                           BackendErrorKind::kDiscovery,
                           "No Metal devices available");
    }

    std::filesystem::path source_path(path);
    std::string source;
    std::string error;
    if (!ReadFile(source_path, &source, &error)) {
        return MetalStatus(StatusCode::kInvalidArgument, BackendErrorKind::kIo,
                           error);
    }

    std::vector<MetalKernel> out;
    std::string last_error;
    for (size_t i = 0; i < devices_.size(); ++i) {
        auto& dev = devices_[i];
        std::string build_options = BuildOptions(dev, extra_build_options);
        std::string cache_key =
            CacheKey(dev, kernel_name, build_options, source);
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
            trace.enabled = exec_config_.trace_kernels;
            std::string trace_path;
            if (TraceKernelSource(trace, &trace_path)) {
                LogBackend({LogLevel::kTrace, BackendType::kMetal,
                            BackendErrorKind::kBuild, "kernel trace written",
                            "build", dev.desc.index, dev.desc.name, 0, "",
                            trace_path});
            }

            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            auto macro_defs = ParseDefineOptions(build_options);
            NSMutableDictionary<NSString*, NSObject*>* macros =
                [NSMutableDictionary dictionaryWithCapacity:macro_defs.size()];
            for (const auto& kv : macro_defs) {
                NSString* key =
                    [NSString stringWithUTF8String:kv.first.c_str()];
                NSString* value =
                    kv.second.empty()
                        ? @"1"
                        : [NSString stringWithUTF8String:kv.second.c_str()];
                if (key) {
                    macros[key] = value;
                }
            }
            options.preprocessorMacros = macros;
            NSError* ns_error = nil;
            id<MTLLibrary> library = [dev.device
                newLibraryWithSource:[NSString
                                         stringWithUTF8String:source.c_str()]
                             options:options
                               error:&ns_error];
            if (!library) {
                std::string build_log;
                int64_t error_code = 0;
                std::string error_name;
                if (ns_error) {
                    error_code = static_cast<int64_t>(ns_error.code);
                    if (ns_error.domain) {
                        error_name = ns_error.domain.UTF8String;
                    }
                    if (ns_error.localizedDescription) {
                        build_log = ns_error.localizedDescription.UTF8String;
                    }
                }
                last_error = build_log.empty()
                                 ? "Metal library compilation failed"
                                 : build_log;
                LogBackend({LogLevel::kWarn, BackendType::kMetal,
                            BackendErrorKind::kBuild,
                            "Metal library compilation failed", "build",
                            dev.desc.index, dev.desc.name, error_code,
                            error_name, "", build_log});
                continue;
            }
            id<MTLFunction> function = [library
                newFunctionWithName:[NSString
                                        stringWithUTF8String:kernel_name
                                                                 .c_str()]];
            if (!function) {
                last_error = "Metal function not found";
                LogBackend({LogLevel::kWarn, BackendType::kMetal,
                            BackendErrorKind::kBuild, last_error, "build",
                            dev.desc.index, dev.desc.name});
                continue;
            }
            pipeline =
                [dev.device newComputePipelineStateWithFunction:function
                                                          error:&ns_error];
            if (!pipeline) {
                last_error = ns_error ? ns_error.localizedDescription.UTF8String
                                      : "Metal pipeline creation failed";
                LogBackend({LogLevel::kWarn, BackendType::kMetal,
                            BackendErrorKind::kBuild, last_error, "build",
                            dev.desc.index, dev.desc.name});
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
        if (last_error.empty())
            last_error = "No Metal kernels built";
        return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kBuild,
                           last_error);
    }
    return out;
}

Status MetalBackend::ReleaseKernel(MetalKernel* kernel) const {
    if (!kernel)
        return Status::OK();
    kernel->pipeline = nullptr;
    return Status::OK();
}

Status MetalBackend::LaunchKernel(
    const MetalKernel& kernel,
    const MetalLaunchConfig& config,
    const std::vector<MetalKernelArg>& args) const {
    ProfilingScope scope(profiling_.get(),
                         {ProfilingEventKind::kKernelLaunch,
                          BackendType::kMetal, kernel.device_index});
    scope.SetLabel(kernel.name);
    Status status = EnsureInitialized();
    if (!status.ok()) {
        scope.SetStatus(status.code);
        return status;
    }
    if (kernel.device_index < 0 ||
        kernel.device_index >= static_cast<int>(devices_.size())) {
        Status err = MetalStatus(StatusCode::kInvalidArgument,
                                 BackendErrorKind::kInvalidArgument,
                                 "Invalid Metal device index");
        scope.SetStatus(err.code);
        return err;
    }
    auto& dev = devices_[kernel.device_index];
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)kernel.pipeline;
    if (!pipeline) {
        Status err = MetalStatus(StatusCode::kInvalidArgument,
                                 BackendErrorKind::kInvalidArgument,
                                 "Invalid Metal pipeline");
        scope.SetStatus(err.code);
        return err;
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
    MTLSize threads =
        MTLSizeMake(config.threads[0], config.threads[1], config.threads[2]);
    if (config.use_threads) {
        [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    } else {
        NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
        NSUInteger width =
            std::min(static_cast<NSUInteger>(config.grid[0]), max_threads);
        MTLSize threads_per_group = MTLSizeMake(width, 1, 1);
        NSUInteger groups = (config.grid[0] + width - 1) / width;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:threads_per_group];
    }

    [enc endEncoding];
    [cmd commit];
    ExecutionConfig exec_config = GetExecutionConfig();
    if (exec_config.sync_on_launch) {
        if (exec_config.kernel_timeout_ms > 0) {
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::milliseconds(exec_config.kernel_timeout_ms);
            while (true) {
                MTLCommandBufferStatus status = cmd.status;
                if (status == MTLCommandBufferStatusCompleted) {
                    break;
                }
                if (status == MTLCommandBufferStatusError) {
                    Status err = MetalErrorStatus(
                        StatusCode::kInternal, BackendErrorKind::kRuntime,
                        "command buffer failed", cmd.error);
                    scope.SetStatus(err.code);
                    return err;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    Status st = MetalStatus(
                        StatusCode::kUnavailable, BackendErrorKind::kRuntime,
                        "kernel timeout after " +
                            std::to_string(exec_config.kernel_timeout_ms) +
                            " ms");
                    st.backend_code = exec_config.kernel_timeout_ms;
                    st.backend_error_name = "timeout";
                    scope.SetStatus(st.code);
                    return st;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else {
            [cmd waitUntilCompleted];
        }
        if (cmd.error) {
            Status err = MetalErrorStatus(StatusCode::kInternal,
                                          BackendErrorKind::kRuntime,
                                          "command buffer failed", cmd.error);
            scope.SetStatus(err.code);
            return err;
        }
    }
    return Status::OK();
}

Status MetalBackend::SmokeTest() const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;

    std::string kernel_dir = KernelDir();
    if (kernel_dir.empty()) {
        return MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kIo,
                           "Metal kernel directory not found");
    }

    const std::string kernel_path =
        (std::filesystem::path(kernel_dir) / "lattice_smoke.metal").string();
    auto add_kernels_or = BuildKernelsFromFile(kernel_path, "vec_add");
    if (!add_kernels_or.ok())
        return add_kernels_or.status();
    auto add_kernels = add_kernels_or.value();
    if (add_kernels.empty()) {
        return MetalStatus(StatusCode::kUnavailable,
                           BackendErrorKind::kDiscovery,
                           "No Metal devices available");
    }
    auto mul_kernels_or = BuildKernelsFromFile(kernel_path, "vec_mul");
    if (!mul_kernels_or.ok()) {
        for (auto& kernel : add_kernels) {
            ReleaseKernel(&kernel);
        }
        return mul_kernels_or.status();
    }
    auto mul_kernels = mul_kernels_or.value();
    std::unordered_map<int, MetalKernel> mul_by_device;
    for (auto& kernel : mul_kernels) {
        mul_by_device[kernel.device_index] = kernel;
    }

    constexpr size_t kCount = 1024;
    std::vector<float> a(kCount, 1.25f);
    std::vector<float> b(kCount, 2.5f);
    std::vector<float> out(kCount, 0.0f);

    for (const auto& kernel : add_kernels) {
        auto it = mul_by_device.find(kernel.device_index);
        if (it == mul_by_device.end()) {
            for (auto& add_kernel : add_kernels) {
                ReleaseKernel(&add_kernel);
            }
            for (auto& mul_kernel : mul_kernels) {
                ReleaseKernel(&mul_kernel);
            }
            return MetalStatus(StatusCode::kUnavailable,
                               BackendErrorKind::kDiscovery,
                               "Missing Metal vec_mul kernel");
        }
        const MetalKernel& mul_kernel = it->second;
        auto buf_a_or =
            CreateBuffer(kernel.device_index, kCount * sizeof(float));
        if (!buf_a_or.ok())
            return buf_a_or.status();
        auto buf_b_or =
            CreateBuffer(kernel.device_index, kCount * sizeof(float));
        if (!buf_b_or.ok())
            return buf_b_or.status();
        auto buf_out_or =
            CreateBuffer(kernel.device_index, kCount * sizeof(float));
        if (!buf_out_or.ok())
            return buf_out_or.status();

        auto buf_a = buf_a_or.value();
        auto buf_b = buf_b_or.value();
        auto buf_out = buf_out_or.value();

        status = WriteBuffer(kernel.device_index, buf_a, a.data(),
                             a.size() * sizeof(float));
        if (!status.ok())
            return status;
        status = WriteBuffer(kernel.device_index, buf_b, b.data(),
                             b.size() * sizeof(float));
        if (!status.ok())
            return status;

        MetalLaunchConfig cfg;
        cfg.grid[0] = static_cast<uint32_t>(kCount);
        std::vector<MetalKernelArg> args;
        args.push_back(MetalKernelArg::Buffer(buf_a.handle));
        args.push_back(MetalKernelArg::Buffer(buf_b.handle));
        args.push_back(MetalKernelArg::Buffer(buf_out.handle));
        unsigned int count = static_cast<unsigned int>(kCount);
        args.push_back(MetalKernelArg::Value(&count, sizeof(count)));

        status = LaunchKernel(kernel, cfg, args);
        if (!status.ok())
            return status;

        status = ReadBuffer(kernel.device_index, buf_out, out.data(),
                            out.size() * sizeof(float));
        if (!status.ok())
            return status;

        for (size_t i = 0; i < kCount; ++i) {
            if (out[i] != a[i] + b[i]) {
                return MetalStatus(StatusCode::kInternal,
                                   BackendErrorKind::kRuntime,
                                   "Metal smoke test failed: incorrect output");
            }
        }

        status = LaunchKernel(mul_kernel, cfg, args);
        if (!status.ok())
            return status;

        status = ReadBuffer(kernel.device_index, buf_out, out.data(),
                            out.size() * sizeof(float));
        if (!status.ok())
            return status;

        for (size_t i = 0; i < kCount; ++i) {
            if (out[i] != a[i] * b[i]) {
                return MetalStatus(StatusCode::kInternal,
                                   BackendErrorKind::kRuntime,
                                   "Metal smoke test failed: vec_mul mismatch");
            }
        }

        ReleaseBuffer(&buf_a);
        ReleaseBuffer(&buf_b);
        ReleaseBuffer(&buf_out);
    }

    for (auto& kernel : add_kernels) {
        ReleaseKernel(&kernel);
    }
    for (auto& kernel : mul_kernels) {
        ReleaseKernel(&kernel);
    }

    return Status::OK();
}

Status MetalBackend::EnsureInitialized() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (initialized_)
        return init_status_;
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

        NSString* os_version =
            [[NSProcessInfo processInfo] operatingSystemVersionString];
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

        DeviceSelectionOptions selection =
            LoadDeviceSelectionOptions("LATTICE_METAL");
        DeviceSelectionResult selected = SelectDevices(identities, selection);
        if (selected.indices.empty()) {
            init_status_ = MetalStatus(
                StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                selected.diagnostics.empty() ? "No Metal devices selected"
                                             : selected.diagnostics);
            return init_status_;
        }

        for (int idx : selected.indices) {
            if (idx < 0 || idx >= static_cast<int>(device_list.size()))
                continue;
            id<MTLDevice> device = device_list[static_cast<size_t>(idx)];
            DeviceContext ctx;
            ctx.device = device;
            ctx.desc.index = idx;
            ctx.desc.name = device.name.UTF8String;
            ctx.desc.vendor = "Apple";
            ctx.desc.driver_version = runtime_str;
            ctx.desc.runtime_version = runtime_str;
            ctx.desc.max_threadgroup_size =
                device.maxThreadsPerThreadgroup.width;
            ctx.desc.shared_mem_bytes = device.maxThreadgroupMemoryLength;
            ctx.caps.is_gpu = true;
            ctx.caps.local_mem_bytes = ctx.desc.shared_mem_bytes;
            ctx.caps.max_work_item_sizes[0] =
                device.maxThreadsPerThreadgroup.width;
            ctx.caps.max_work_item_sizes[1] =
                device.maxThreadsPerThreadgroup.height;
            ctx.caps.max_work_item_sizes[2] =
                device.maxThreadsPerThreadgroup.depth;
            ctx.caps.max_work_group_size =
                device.maxThreadsPerThreadgroup.width *
                device.maxThreadsPerThreadgroup.height *
                device.maxThreadsPerThreadgroup.depth;
            ctx.caps.max_threads_per_block = ctx.caps.max_work_group_size;
            bool feature = false;
            if (QueryBoolSelector(device, @selector(supports64BitFloat),
                                  &feature)) {
                ctx.caps.fp64 =
                    feature ? CapabilityStatus::kYes : CapabilityStatus::kNo;
            }
            if (QueryBoolSelector(device, @selector(supports16BitFloat),
                                  &feature)) {
                ctx.caps.fp16 =
                    feature ? CapabilityStatus::kYes : CapabilityStatus::kNo;
            }
            ctx.caps.quirks =
                QueryDeviceQuirks(BackendType::kMetal, ctx.desc.vendor,
                                  ctx.desc.name, runtime_str);
            ctx.caps.is_software =
                (ctx.caps.quirks.flags & kSoftwareEmulation) != 0;
            if (ctx.caps.quirks.flags & kDisableFp16)
                ctx.caps.fp16 = CapabilityStatus::kNo;
            if (ctx.caps.quirks.flags & kDisableFp64)
                ctx.caps.fp64 = CapabilityStatus::kNo;

            DeviceMetadata meta = BuildDeviceMetadata(ctx.desc, ctx.caps);
            ctx.fingerprint = DeviceFingerprint(meta);

            if (ctx.caps.quirks.disabled) {
                LogBackend({LogLevel::kWarn, BackendType::kMetal,
                            BackendErrorKind::kDiscovery,
                            "skipping device: " + ctx.caps.quirks.reason,
                            "device_skip", ctx.desc.index, ctx.desc.name});
                continue;
            }
            ctx.queue = [device newCommandQueue];
            if (!ctx.queue) {
                LogBackend({LogLevel::kWarn, BackendType::kMetal,
                            BackendErrorKind::kContext, "queue init failed",
                            "queue", ctx.desc.index, ctx.desc.name});
                continue;
            }
            devices_.push_back(std::move(ctx));
        }
    }

    if (devices_.empty()) {
        init_status_ =
            MetalStatus(StatusCode::kUnavailable, BackendErrorKind::kContext,
                        "No Metal devices initialized");
        return init_status_;
    }

    {
        DeviceMetadataStore meta_store;
        std::string meta_error;
        for (const auto& dev : devices_) {
            const DeviceMetadata meta = BuildDeviceMetadata(dev.desc, dev.caps);
            std::string previous_fingerprint;
            if (!meta_store.Write(meta, &meta_error, &previous_fingerprint)) {
                LogBackend({LogLevel::kWarn, BackendType::kMetal,
                            BackendErrorKind::kIo,
                            "metadata persist failed: " + meta_error,
                            "metadata", dev.desc.index, dev.desc.name});
                meta_error.clear();
            }
        }
    }

    for (size_t i = 0; i < devices_.size(); ++i) {
        const auto& dev = devices_[i];
        LogRecord record;
        record.level = LogLevel::kInfo;
        record.backend = BackendType::kMetal;
        record.kind = BackendErrorKind::kDiscovery;
        record.message = dev.desc.name;
        record.operation = "device_info";
        record.device_index = dev.desc.index;
        record.device_name = dev.desc.name;
        record.device_info = MetalDeviceInfoJson(dev.desc, dev.caps);
        LogBackend(record);
    }

    init_status_ = Status::OK();
    return init_status_;
}

MemoryPool* MetalBackend::DevicePool(int device_index) const {
    if (device_index < 0 || device_index >= static_cast<int>(devices_.size()))
        return nullptr;
    std::lock_guard<std::mutex> lock(mu_);
    auto& dev = devices_[device_index];
    if (dev.device_pool)
        return dev.device_pool.get();

    MemoryPoolConfig config = MetalDevicePoolConfig();
    const int idx = device_index;
    auto alloc_fn = [this, idx](size_t bytes,
                                size_t alignment) -> StatusOr<PoolBlock> {
        (void)alignment;
        auto& device = devices_[idx];
        id<MTLBuffer> buffer =
            [device.device newBufferWithLength:bytes
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
        if (!block.handle)
            return Status::OK();
        CFRelease(reinterpret_cast<void*>(block.handle));
        return Status::OK();
    };
    auto scrub_fn = [this, idx](const PoolBlock& block) -> Status {
        if (!block.handle || block.bytes == 0)
            return Status::OK();
        id<MTLBuffer> buffer =
            (__bridge id<MTLBuffer>)reinterpret_cast<void*>(block.handle);
        void* dst = buffer.contents;
        if (dst) {
            std::memset(dst, 0, block.bytes);
        }
        return Status::OK();
    };

    std::ostringstream label;
    label << "metal_device_pool_" << dev.desc.index;
    dev.device_pool = std::make_unique<MemoryPool>(label.str(), config,
                                                   alloc_fn, free_fn, scrub_fn);
    return dev.device_pool.get();
}

MemoryPool* MetalBackend::PinnedPool(int device_index) const {
    if (device_index < 0 || device_index >= static_cast<int>(devices_.size()))
        return nullptr;
    std::lock_guard<std::mutex> lock(mu_);
    auto& dev = devices_[device_index];
    if (dev.pinned_pool)
        return dev.pinned_pool.get();

    MemoryPoolConfig config = MetalPinnedPoolConfig();
    const int idx = device_index;
    auto alloc_fn = [this, idx](size_t bytes,
                                size_t alignment) -> StatusOr<PoolBlock> {
        (void)alignment;
        auto& device = devices_[idx];
        MTLResourceOptions options =
            MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
        id<MTLBuffer> buffer = [device.device newBufferWithLength:bytes
                                                          options:options];
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
        if (!block.handle)
            return Status::OK();
        CFRelease(reinterpret_cast<void*>(block.handle));
        return Status::OK();
    };
    auto scrub_fn =
        [secure = config.secure_scrub](const PoolBlock& block) -> Status {
        if (block.host_ptr && block.bytes > 0) {
            if (secure) {
                SecureZero(block.host_ptr, block.bytes);
            } else {
                std::memset(block.host_ptr, 0, block.bytes);
            }
        }
        return Status::OK();
    };

    std::ostringstream label;
    label << "metal_pinned_pool_" << dev.desc.index;
    dev.pinned_pool = std::make_unique<MemoryPool>(label.str(), config,
                                                   alloc_fn, free_fn, scrub_fn);
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

std::string MetalBackend::BuildOptions(const DeviceContext& dev,
                                       const std::string& extra) const {
    std::string options;
    const std::string kernel_dir = KernelDir();
    AppendOption(&options, BuildIncludeOption(kernel_dir, "-I"));
    KernelBuildDefines defs;
    defs.backend = BackendType::kMetal;
    defs.device_index = dev.desc.index;
    defs.abi_version = metal::kAbiVersion;
    defs.abi_version_min = metal::kAbiVersionMin;
    defs.has_fp16 = dev.caps.fp16 == CapabilityStatus::kYes;
    defs.has_fp64 = dev.caps.fp64 == CapabilityStatus::kYes;
    defs.fast_math = exec_config_.enable_fast_math;
    defs.vectorize = exec_config_.enable_vectorize;
    defs.mixed_precision = exec_config_.enable_mixed_precision;
    defs.vendor_id = 0x106B;
    AppendOption(&options, KernelDefineString(defs, "-D"));
    AppendOption(&options, LoadBuildOptionsEnv("LATTICE_METAL_BUILD_OPTIONS"));
    AppendOption(&options, extra);
    if (exec_config_.enable_fast_math) {
        AppendOption(&options, "-ffast-math");
    }
    if (KernelDebugEnabled("LATTICE_METAL_BUILD_DEBUG")) {
        AppendOption(&options, KernelDebugOptions(BackendType::kMetal));
    }
    return options;
}

std::string MetalBackend::CacheKey(const DeviceContext& dev,
                                   const std::string& kernel_name,
                                   const std::string& build_options,
                                   const std::string& source) const {
    std::string meta = dev.fingerprint;
    meta += "|";
    meta += kernel_name;
    meta += "|";
    meta += build_options;
    const std::string meta_hash = Sha256Hex(meta);
    const std::string src_hash = Sha256Hex(source);
    return "metal_" + meta_hash + "_" + src_hash;
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
