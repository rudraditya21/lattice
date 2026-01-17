#include "runtime/backends/opencl_backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
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
#include "runtime/backends/gpu/opencl_loader.h"
#include "runtime/backends/kernel_build.h"
#include "runtime/backends/memory_pool.h"
#include "runtime/backends/memory_stats.h"
#include "runtime/backends/memory_utils.h"
#include "runtime/backends/opencl_abi.h"
#include "runtime/backends/profiling.h"

namespace lattice::runtime {

namespace {

using gpu::OpenCLLoader;

constexpr size_t kMaxPlatforms = 16;

bool HasExtension(const std::string& extensions, const char* needle) {
    return extensions.find(needle) != std::string::npos;
}

DeviceKind DeviceKindFromType(cl_device_type type) {
    if (type & CL_DEVICE_TYPE_CPU)
        return DeviceKind::kCPU;
    if (type & CL_DEVICE_TYPE_GPU)
        return DeviceKind::kGPU;
    if (type & CL_DEVICE_TYPE_ACCELERATOR)
        return DeviceKind::kAccelerator;
    return DeviceKind::kAny;
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

std::string DeviceTypeString(cl_device_type type) {
    if (type & CL_DEVICE_TYPE_GPU)
        return "GPU";
    if (type & CL_DEVICE_TYPE_CPU)
        return "CPU";
    if (type & CL_DEVICE_TYPE_ACCELERATOR)
        return "ACCELERATOR";
    return "UNKNOWN";
}

std::string ProgramBuildLog(const OpenCLLoader& loader,
                            cl_program program,
                            cl_device_id device) {
    size_t log_size = 0;
    loader.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0,
                                 nullptr, &log_size);
    if (log_size <= 1)
        return "";
    std::string log(log_size, '\0');
    loader.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                 log_size, log.data(), nullptr);
    if (!log.empty() && log.back() == '\0')
        log.pop_back();
    return log;
}

Status OpenclStatus(StatusCode code,
                    BackendErrorKind kind,
                    const std::string& message) {
    return MakeBackendError(code, BackendType::kOpenCL, kind, message);
}

Status OpenclErrorStatus(StatusCode code,
                         BackendErrorKind kind,
                         const std::string& message,
                         cl_int err) {
    Status st =
        OpenclStatus(code, kind, message + ": " + gpu::OpenCLErrorString(err));
    st.backend_code = static_cast<int64_t>(err);
    st.backend_error_name = gpu::OpenCLErrorString(err);
    return st;
}

std::string OpenclDeviceInfoJson(const OpenCLDeviceDesc& desc,
                                 const DeviceCapabilities& caps) {
    std::ostringstream out;
    out << "{";
    out << "\"name\":\"" << EscapeJson(desc.name) << "\"";
    if (!desc.vendor.empty()) {
        out << ",\"vendor\":\"" << EscapeJson(desc.vendor) << "\"";
    }
    if (!desc.platform_name.empty()) {
        out << ",\"platform_name\":\"" << EscapeJson(desc.platform_name)
            << "\"";
    }
    if (!desc.platform_vendor.empty()) {
        out << ",\"platform_vendor\":\"" << EscapeJson(desc.platform_vendor)
            << "\"";
    }
    if (!desc.platform_version.empty()) {
        out << ",\"platform_version\":\"" << EscapeJson(desc.platform_version)
            << "\"";
    }
    if (!desc.device_version.empty()) {
        out << ",\"device_version\":\"" << EscapeJson(desc.device_version)
            << "\"";
    }
    if (!desc.runtime_version.empty()) {
        out << ",\"runtime_version\":\"" << EscapeJson(desc.runtime_version)
            << "\"";
    }
    if (!desc.driver_version.empty()) {
        out << ",\"driver_version\":\"" << EscapeJson(desc.driver_version)
            << "\"";
    }
    out << ",\"device_type\":\"" << DeviceTypeString(desc.type) << "\"";
    out << ",\"vendor_id\":" << desc.vendor_id;
    out << ",\"compute_units\":" << desc.compute_units;
    out << ",\"max_clock_mhz\":" << desc.max_clock_mhz;
    out << ",\"local_mem_bytes\":" << desc.local_mem_size;
    out << ",\"max_work_group_size\":" << desc.max_work_group_size;
    out << ",\"fp16\":" << CapabilityToInt(caps.fp16);
    out << ",\"fp64\":" << CapabilityToInt(caps.fp64);
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

DeviceMetadata BuildDeviceMetadata(const OpenCLDeviceDesc& desc,
                                   const DeviceCapabilities& caps) {
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
    if (pos == std::string::npos)
        return "";
    std::string v = version.substr(pos + 9);
    if (v.rfind("3.0", 0) == 0)
        return "-cl-std=CL3.0";
    if (v.rfind("2.2", 0) == 0)
        return "-cl-std=CL2.2";
    if (v.rfind("2.1", 0) == 0)
        return "-cl-std=CL2.1";
    if (v.rfind("2.0", 0) == 0)
        return "-cl-std=CL2.0";
    if (v.rfind("1.2", 0) == 0)
        return "-cl-std=CL1.2";
    if (v.rfind("1.1", 0) == 0)
        return "-cl-std=CL1.1";
    return "";
}

MemoryPoolConfig OpenclDevicePoolConfig() {
    static MemoryPoolConfig config = [] {
        MemoryPoolConfig base = DefaultDevicePoolConfig();
        base = LoadMemoryPoolConfig("LATTICE_DEVICE_POOL", base);
        base = LoadMemoryPoolConfig("LATTICE_OPENCL_DEVICE_POOL", base);
        return base;
    }();
    return config;
}

MemoryPoolConfig OpenclPinnedPoolConfig() {
    static MemoryPoolConfig config = [] {
        MemoryPoolConfig base = DefaultPinnedPoolConfig();
        base = LoadMemoryPoolConfig("LATTICE_PINNED_POOL", base);
        base = LoadMemoryPoolConfig("LATTICE_OPENCL_PINNED_POOL", base);
        return base;
    }();
    return config;
}

Status FillBufferZero(const OpenCLLoader* loader,
                      cl_command_queue queue,
                      cl_mem mem,
                      size_t bytes) {
    if (!loader || !queue || !mem || bytes == 0)
        return Status::OK();
    if (loader->clEnqueueFillBuffer) {
        const cl_uint zero = 0;
        cl_int err = loader->clEnqueueFillBuffer(
            queue, mem, &zero, sizeof(zero), 0, bytes, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            return OpenclStatus(
                StatusCode::kInternal, BackendErrorKind::kMemory,
                "clEnqueueFillBuffer failed: " + gpu::OpenCLErrorString(err));
        }
        loader->clFinish(queue);
        return Status::OK();
    }
    const size_t chunk = 4096;
    std::vector<unsigned char> zeros(chunk, 0);
    size_t offset = 0;
    while (offset < bytes) {
        size_t to_write = std::min(chunk, bytes - offset);
        cl_int err =
            loader->clEnqueueWriteBuffer(queue, mem, CL_FALSE, offset, to_write,
                                         zeros.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            return OpenclStatus(
                StatusCode::kInternal, BackendErrorKind::kMemory,
                "clEnqueueWriteBuffer failed: " + gpu::OpenCLErrorString(err));
        }
        offset += to_write;
    }
    loader->clFinish(queue);
    return Status::OK();
}

class OpenCLEvent final : public Event {
   public:
    OpenCLEvent(const OpenCLLoader* loader,
                cl_command_queue queue,
                ProfilingState* profiling,
                int device_index)
        : loader_(loader),
          queue_(queue),
          profiling_(profiling),
          device_index_(device_index) {}
    ~OpenCLEvent() override {
        if (event_ && loader_ && loader_->clReleaseEvent) {
            loader_->clReleaseEvent(event_);
        }
    }

    void Record() override {
        ProfilingScope scope(profiling_, {ProfilingEventKind::kEventRecord,
                                          BackendType::kOpenCL, device_index_});
        ready_.store(false, std::memory_order_relaxed);
        if (!loader_ || !queue_) {
            scope.SetStatus(StatusCode::kUnavailable);
            ready_.store(true, std::memory_order_relaxed);
            return;
        }
        if (event_ && loader_->clReleaseEvent) {
            loader_->clReleaseEvent(event_);
            event_ = nullptr;
        }
        cl_int err = CL_SUCCESS;
        if (loader_->clEnqueueMarkerWithWaitList) {
            err = loader_->clEnqueueMarkerWithWaitList(queue_, 0, nullptr,
                                                       &event_);
        } else if (loader_->clEnqueueMarker) {
            err = loader_->clEnqueueMarker(queue_, &event_);
        } else {
            loader_->clFinish(queue_);
            ready_.store(true, std::memory_order_relaxed);
            return;
        }
        if (err != CL_SUCCESS) {
            event_ = nullptr;
            ready_.store(true, std::memory_order_relaxed);
            return;
        }
        if (loader_->clFlush) {
            loader_->clFlush(queue_);
        }
    }

    void Wait() override {
        ProfilingScope scope(profiling_, {ProfilingEventKind::kEventWait,
                                          BackendType::kOpenCL, device_index_});
        if (event_ && loader_ && loader_->clWaitForEvents) {
            loader_->clWaitForEvents(1, &event_);
            ready_.store(true, std::memory_order_relaxed);
            return;
        }
        if (loader_ && queue_ && loader_->clFinish) {
            loader_->clFinish(queue_);
        }
        ready_.store(true, std::memory_order_relaxed);
    }

    bool Ready() const override {
        if (event_ && loader_ && loader_->clGetEventInfo) {
            cl_int status = 0;
            loader_->clGetEventInfo(event_, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                    sizeof(status), &status, nullptr);
            return status == CL_COMPLETE;
        }
        return ready_.load(std::memory_order_relaxed);
    }

    cl_event handle() const { return event_; }

   private:
    const OpenCLLoader* loader_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_event event_ = nullptr;
    std::atomic<bool> ready_{false};
    ProfilingState* profiling_ = nullptr;
    int device_index_ = -1;
};

class OpenCLStream final : public Stream {
   public:
    OpenCLStream(const OpenCLLoader* loader,
                 cl_command_queue queue,
                 bool owns_queue,
                 ProfilingState* profiling,
                 int device_index)
        : loader_(loader),
          queue_(queue),
          owns_queue_(owns_queue),
          profiling_(profiling),
          device_index_(device_index) {}
    ~OpenCLStream() override {
        if (owns_queue_ && loader_ && loader_->clReleaseCommandQueue &&
            queue_) {
            loader_->clReleaseCommandQueue(queue_);
        }
    }
    void Submit(std::function<void()> fn) override {
        for (auto& dep : deps_) {
            dep->Wait();
        }
        deps_.clear();
        fn();
    }
    void Synchronize() override {
        ProfilingScope scope(profiling_, {ProfilingEventKind::kStreamSync,
                                          BackendType::kOpenCL, device_index_});
        if (loader_ && queue_)
            loader_->clFinish(queue_);
        deps_.clear();
    }
    void AddDependency(const std::shared_ptr<Event>& ev) override {
        if (!ev)
            return;
        auto* ocl_event = dynamic_cast<OpenCLEvent*>(ev.get());
        if (ocl_event && loader_ && queue_) {
            cl_event handle = ocl_event->handle();
            if (handle && loader_->clEnqueueBarrierWithWaitList) {
                cl_int err = loader_->clEnqueueBarrierWithWaitList(
                    queue_, 1, &handle, nullptr);
                if (err == CL_SUCCESS)
                    return;
            } else if (handle && loader_->clWaitForEvents) {
                loader_->clWaitForEvents(1, &handle);
                return;
            }
        }
        deps_.push_back(ev);
    }
    StatusOr<std::shared_ptr<Event>> CreateEvent() const override {
        if (!queue_) {
            return OpenclStatus(StatusCode::kUnavailable,
                                BackendErrorKind::kContext,
                                "OpenCL queue unavailable");
        }
        return std::make_shared<OpenCLEvent>(loader_, queue_, profiling_,
                                             device_index_);
    }
    void RecordEvent(const std::shared_ptr<Event>& ev) override {
        if (!ev)
            return;
        Submit([ev]() { ev->Record(); });
    }
    void SetPriority(int priority) override { priority_ = priority; }

   private:
    const OpenCLLoader* loader_ = nullptr;
    cl_command_queue queue_ = nullptr;
    bool owns_queue_ = false;
    int priority_ = 0;
    std::vector<std::shared_ptr<Event>> deps_;
    ProfilingState* profiling_ = nullptr;
    int device_index_ = -1;
};

}  // namespace

struct OpenCLBackend::DeviceContext {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    bool queue_profiling = false;
    OpenCLDeviceDesc desc;
    DeviceCapabilities caps;
    std::string fingerprint;
    std::unordered_map<std::string, cl_program> program_cache;
    std::unordered_map<cl_mem_flags, std::unique_ptr<MemoryPool>> device_pools;
    std::unique_ptr<MemoryPool> pinned_pool;
};

OpenCLBackend::OpenCLBackend() {
    exec_config_ = LoadExecutionConfig("LATTICE", exec_config_);
    exec_config_ = LoadExecutionConfig("LATTICE_OPENCL", exec_config_);
    profiling_ = std::make_shared<ProfilingState>(BackendType::kOpenCL);
    profiling_->SetEnabled(exec_config_.enable_profiling);
}

OpenCLBackend::~OpenCLBackend() {
    if (loader_.Loaded()) {
        for (auto& dev : devices_) {
            MemoryPoolStats device_stats;
            for (auto& entry : dev.device_pools) {
                if (entry.second) {
                    AccumulateMemoryPoolStats(&device_stats,
                                              entry.second->Stats());
                    entry.second->Trim();
                }
            }
            if (HasOutstandingAllocs(device_stats)) {
                LogBackend({LogLevel::kWarn, BackendType::kOpenCL,
                            BackendErrorKind::kMemory,
                            "device pool has outstanding allocations (" +
                                FormatPoolStats(device_stats) + ")",
                            "device_pool_leak", dev.desc.index, dev.desc.name});
            }
            if (dev.pinned_pool) {
                MemoryPoolStats stats = dev.pinned_pool->Stats();
                if (HasOutstandingAllocs(stats)) {
                    LogBackend({LogLevel::kWarn, BackendType::kOpenCL,
                                BackendErrorKind::kMemory,
                                "pinned pool has outstanding allocations (" +
                                    FormatPoolStats(stats) + ")",
                                "pinned_pool_leak", dev.desc.index,
                                dev.desc.name});
                }
                dev.pinned_pool->Trim();
            }
            for (auto& entry : dev.program_cache) {
                if (entry.second)
                    loader_.clReleaseProgram(entry.second);
            }
            dev.program_cache.clear();
            if (dev.queue)
                loader_.clReleaseCommandQueue(dev.queue);
            if (dev.context)
                loader_.clReleaseContext(dev.context);
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
    caps.supports_sparse = true;
    caps.supports_ragged = true;
    caps.supports_fft = true;
    caps.supports_blas = true;
    caps.supports_conv = true;
    caps.supports_rng = false;
    caps.supports_events = true;
    caps.supports_profiling = true;
    caps.supported_dtypes = {DType::kF16, DType::kBF16, DType::kF32,
                             DType::kF64, DType::kI32,  DType::kU32};
    return caps;
}

StatusOr<std::shared_ptr<Stream>> OpenCLBackend::CreateStream() const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    if (devices_.empty()) {
        return OpenclStatus(StatusCode::kUnavailable,
                            BackendErrorKind::kDiscovery,
                            "No OpenCL devices available");
    }
    const auto& dev = devices_[0];
    cl_command_queue queue = nullptr;
    bool owns_queue = false;
    ExecutionConfig config = GetExecutionConfig();
    cl_command_queue_properties props =
        config.enable_profiling ? CL_QUEUE_PROFILING_ENABLE : 0;
    if (loader_.clCreateCommandQueue) {
        cl_int err = CL_SUCCESS;
        queue =
            loader_.clCreateCommandQueue(dev.context, dev.device, props, &err);
        if (err == CL_SUCCESS && queue) {
            owns_queue = true;
        } else {
            queue = nullptr;
        }
    }
    if (!queue) {
        queue = dev.queue;
        owns_queue = false;
    }
    if (!queue) {
        return OpenclStatus(StatusCode::kUnavailable,
                            BackendErrorKind::kContext,
                            "OpenCL queue unavailable");
    }
    return std::make_shared<OpenCLStream>(&loader_, queue, owns_queue,
                                          profiling_.get(), dev.desc.index);
}

StatusOr<std::shared_ptr<Event>> OpenCLBackend::CreateEvent() const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    if (devices_.empty()) {
        return OpenclStatus(StatusCode::kUnavailable,
                            BackendErrorKind::kDiscovery,
                            "No OpenCL devices available");
    }
    return std::make_shared<OpenCLEvent>(
        &loader_, devices_[0].queue, profiling_.get(), devices_[0].desc.index);
}

StatusOr<Allocation> OpenCLBackend::Allocate(size_t bytes,
                                             size_t alignment) const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    if (devices_.empty()) {
        return OpenclStatus(StatusCode::kUnavailable,
                            BackendErrorKind::kDiscovery,
                            "No OpenCL devices available");
    }
    if (bytes == 0) {
        Allocation empty;
        empty.kind = AllocationKind::kDevice;
        return empty;
    }
    auto* pool = DevicePool(0, CL_MEM_READ_WRITE);
    if (!pool) {
        return OpenclStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
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

Status OpenCLBackend::Deallocate(const Allocation& alloc) const {
    if (!alloc.device_handle)
        return Status::OK();
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    auto* pool = DevicePool(0, CL_MEM_READ_WRITE);
    if (!pool) {
        return OpenclStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                            "Device pool unavailable");
    }
    return pool->Release(reinterpret_cast<uintptr_t>(alloc.device_handle));
}

StatusOr<Allocation> OpenCLBackend::AllocatePinned(size_t bytes,
                                                   size_t alignment) const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    if (devices_.empty()) {
        return OpenclStatus(StatusCode::kUnavailable,
                            BackendErrorKind::kDiscovery,
                            "No OpenCL devices available");
    }
    if (bytes == 0) {
        Allocation empty;
        empty.kind = AllocationKind::kPinnedHost;
        return empty;
    }
    auto* pool = PinnedPool(0);
    if (!pool) {
        return OpenclStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
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

Status OpenCLBackend::DeallocatePinned(const Allocation& alloc) const {
    if (!alloc.ptr)
        return Status::OK();
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    auto* pool = PinnedPool(0);
    if (!pool) {
        return OpenclStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                            "Pinned pool unavailable");
    }
    return pool->Release(reinterpret_cast<uintptr_t>(alloc.ptr));
}

int OpenCLBackend::NumThreads() const {
    return 1;
}

size_t OpenCLBackend::OutstandingAllocs() const {
    size_t total = 0;
    for (int i = 0; i < static_cast<int>(devices_.size()); ++i) {
        auto& dev = devices_[i];
        for (const auto& entry : dev.device_pools) {
            total += entry.second ? entry.second->Outstanding() : 0;
        }
        if (dev.pinned_pool)
            total += dev.pinned_pool->Outstanding();
    }
    return total;
}

BackendMemoryStats OpenCLBackend::MemoryStats() const {
    BackendMemoryStats stats;
    Status status = EnsureInitialized();
    if (!status.ok())
        return stats;
    for (int i = 0; i < static_cast<int>(devices_.size()); ++i) {
        auto& dev = devices_[i];
        for (const auto& entry : dev.device_pools) {
            if (entry.second) {
                AccumulateMemoryPoolStats(&stats.device, entry.second->Stats());
            }
        }
        if (dev.pinned_pool) {
            AccumulateMemoryPoolStats(&stats.pinned, dev.pinned_pool->Stats());
        }
    }
    return stats;
}

std::vector<DeviceMemoryStats> OpenCLBackend::MemoryStatsByDevice() const {
    std::vector<DeviceMemoryStats> out;
    Status status = EnsureInitialized();
    if (!status.ok())
        return out;
    out.reserve(devices_.size());
    for (const auto& dev : devices_) {
        DeviceMemoryStats entry;
        entry.backend = BackendType::kOpenCL;
        entry.device_index = dev.desc.index;
        entry.device_name = dev.desc.name;
        for (const auto& pool_entry : dev.device_pools) {
            if (pool_entry.second) {
                AccumulateMemoryPoolStats(&entry.device,
                                          pool_entry.second->Stats());
            }
        }
        if (dev.pinned_pool) {
            entry.pinned = dev.pinned_pool->Stats();
        }
        out.push_back(std::move(entry));
    }
    return out;
}

ExecutionConfig OpenCLBackend::GetExecutionConfig() const {
    std::lock_guard<std::mutex> lock(config_mu_);
    return exec_config_;
}

Status OpenCLBackend::SetExecutionConfig(const ExecutionConfig& config) {
    bool profiling_changed = false;
    {
        std::lock_guard<std::mutex> lock(config_mu_);
        profiling_changed =
            exec_config_.enable_profiling != config.enable_profiling;
        exec_config_ = config;
    }
    if (profiling_) {
        profiling_->SetEnabled(exec_config_.enable_profiling);
    }
    if (!profiling_changed)
        return Status::OK();
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    std::lock_guard<std::mutex> lock(mu_);
    cl_command_queue_properties props =
        config.enable_profiling ? CL_QUEUE_PROFILING_ENABLE : 0;
    for (auto& dev : devices_) {
        if (dev.queue && loader_.clReleaseCommandQueue) {
            loader_.clReleaseCommandQueue(dev.queue);
            dev.queue = nullptr;
        }
        cl_int err = CL_SUCCESS;
        dev.queue =
            loader_.clCreateCommandQueue(dev.context, dev.device, props, &err);
        if (err != CL_SUCCESS || !dev.queue) {
            dev.queue = nullptr;
            return OpenclStatus(
                StatusCode::kUnavailable, BackendErrorKind::kContext,
                "queue reinit failed: " + gpu::OpenCLErrorString(err));
        }
        dev.queue_profiling = config.enable_profiling;
    }
    return Status::OK();
}

ProfilingCounters OpenCLBackend::ProfilingStats() const {
    if (!profiling_) {
        return ProfilingCounters{};
    }
    return profiling_->Snapshot();
}

void OpenCLBackend::ResetProfilingStats() {
    if (profiling_) {
        profiling_->Reset();
    }
}

void OpenCLBackend::SetProfilingHook(ProfilingHook hook) {
    if (profiling_) {
        profiling_->SetHook(std::move(hook));
    }
}

StatusOr<uint64_t> OpenCLBackend::ElapsedNs(
    const std::shared_ptr<Event>& start,
    const std::shared_ptr<Event>& end) const {
    if (!start || !end) {
        return Status::Invalid("Missing events for elapsed time");
    }
    ExecutionConfig config = GetExecutionConfig();
    if (!config.enable_profiling) {
        return Status::Unavailable("OpenCL profiling disabled");
    }
    auto* start_ev = dynamic_cast<OpenCLEvent*>(start.get());
    auto* end_ev = dynamic_cast<OpenCLEvent*>(end.get());
    if (!start_ev || !end_ev) {
        return Status::Invalid("Events do not belong to OpenCL backend");
    }
    if (!loader_.clGetEventProfilingInfo) {
        return Status::Unavailable("OpenCL profiling info not available");
    }
    cl_event start_handle = start_ev->handle();
    cl_event end_handle = end_ev->handle();
    if (!start_handle || !end_handle) {
        return Status::Invalid("Events have no OpenCL handles");
    }
    if (loader_.clWaitForEvents) {
        loader_.clWaitForEvents(1, &end_handle);
    }
    cl_ulong start_time = 0;
    cl_ulong end_time = 0;
    cl_int err = loader_.clGetEventProfilingInfo(
        start_handle, CL_PROFILING_COMMAND_START, sizeof(start_time),
        &start_time, nullptr);
    if (err != CL_SUCCESS) {
        return OpenclStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                            "clGetEventProfilingInfo start failed: " +
                                gpu::OpenCLErrorString(err));
    }
    err = loader_.clGetEventProfilingInfo(end_handle, CL_PROFILING_COMMAND_END,
                                          sizeof(end_time), &end_time, nullptr);
    if (err != CL_SUCCESS) {
        return OpenclStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                            "clGetEventProfilingInfo end failed: " +
                                gpu::OpenCLErrorString(err));
    }
    if (end_time < start_time) {
        return Status::Invalid("OpenCL event timestamps invalid");
    }
    return static_cast<uint64_t>(end_time - start_time);
}

void OpenCLBackend::SetDefaultPriority(int priority) {
    default_priority_ = priority;
}

void OpenCLBackend::SetDeterministic(bool deterministic) {
    deterministic_ = deterministic;
}

int OpenCLBackend::DeviceCount() const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return 0;
    return static_cast<int>(devices_.size());
}

std::vector<OpenCLDeviceDesc> OpenCLBackend::DeviceInfo() const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return {};
    std::vector<OpenCLDeviceDesc> out;
    for (const auto& dev : devices_) {
        out.push_back(dev.desc);
    }
    return out;
}

std::vector<DeviceCapabilities> OpenCLBackend::DeviceCaps() const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return {};
    std::vector<DeviceCapabilities> out;
    for (const auto& dev : devices_) {
        out.push_back(dev.caps);
    }
    return out;
}

StatusOr<OpenCLBuffer> OpenCLBackend::CreateBuffer(int device_index,
                                                   size_t bytes,
                                                   cl_mem_flags flags) const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
        return OpenclStatus(StatusCode::kInvalidArgument,
                            BackendErrorKind::kInvalidArgument,
                            "Invalid device index");
    }
    if (bytes == 0) {
        return OpenCLBuffer{nullptr, 0, flags, device_index};
    }
    auto* pool = DevicePool(device_index, flags);
    if (!pool) {
        return OpenclStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                            "Device pool unavailable");
    }
    auto block_or = pool->Acquire(bytes, 64);
    if (!block_or.ok())
        return block_or.status();
    auto block = block_or.value();
    return OpenCLBuffer{reinterpret_cast<cl_mem>(block.handle), bytes, flags,
                        device_index};
}

Status OpenCLBackend::ReleaseBuffer(OpenCLBuffer* buffer) const {
    if (!buffer || !buffer->mem)
        return Status::OK();
    auto* pool = DevicePool(buffer->device_index, buffer->flags);
    if (!pool) {
        return OpenclStatus(StatusCode::kUnavailable, BackendErrorKind::kMemory,
                            "Device pool unavailable");
    }
    Status status = pool->Release(reinterpret_cast<uintptr_t>(buffer->mem));
    if (!status.ok())
        return status;
    buffer->mem = nullptr;
    buffer->bytes = 0;
    buffer->flags = CL_MEM_READ_WRITE;
    buffer->device_index = -1;
    return Status::OK();
}

Status OpenCLBackend::WriteBuffer(int device_index,
                                  const OpenCLBuffer& buffer,
                                  const void* data,
                                  size_t bytes,
                                  size_t offset) const {
    ProfilingScope scope(
        profiling_.get(),
        {ProfilingEventKind::kMemcpyH2D, BackendType::kOpenCL, device_index});
    scope.SetBytes(bytes);
    Status status = EnsureInitialized();
    if (!status.ok()) {
        scope.SetStatus(status.code);
        return status;
    }
    if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
        Status err = OpenclStatus(StatusCode::kInvalidArgument,
                                  BackendErrorKind::kInvalidArgument,
                                  "Invalid device index");
        scope.SetStatus(err.code);
        return err;
    }
    if (!buffer.mem) {
        Status err = OpenclStatus(StatusCode::kInvalidArgument,
                                  BackendErrorKind::kInvalidArgument,
                                  "Invalid buffer handle");
        scope.SetStatus(err.code);
        return err;
    }
    if (bytes + offset > buffer.bytes) {
        Status err = OpenclStatus(StatusCode::kInvalidArgument,
                                  BackendErrorKind::kInvalidArgument,
                                  "Write exceeds buffer size");
        scope.SetStatus(err.code);
        return err;
    }
    cl_int err = loader_.clEnqueueWriteBuffer(devices_[device_index].queue,
                                              buffer.mem, CL_TRUE, offset,
                                              bytes, data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        Status st =
            OpenclErrorStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                              "clEnqueueWriteBuffer failed", err);
        scope.SetStatus(st.code);
        return st;
    }
    return Status::OK();
}

Status OpenCLBackend::ReadBuffer(int device_index,
                                 const OpenCLBuffer& buffer,
                                 void* data,
                                 size_t bytes,
                                 size_t offset) const {
    ProfilingScope scope(
        profiling_.get(),
        {ProfilingEventKind::kMemcpyD2H, BackendType::kOpenCL, device_index});
    scope.SetBytes(bytes);
    Status status = EnsureInitialized();
    if (!status.ok()) {
        scope.SetStatus(status.code);
        return status;
    }
    if (device_index < 0 || device_index >= static_cast<int>(devices_.size())) {
        Status err = OpenclStatus(StatusCode::kInvalidArgument,
                                  BackendErrorKind::kInvalidArgument,
                                  "Invalid device index");
        scope.SetStatus(err.code);
        return err;
    }
    if (!buffer.mem) {
        Status err = OpenclStatus(StatusCode::kInvalidArgument,
                                  BackendErrorKind::kInvalidArgument,
                                  "Invalid buffer handle");
        scope.SetStatus(err.code);
        return err;
    }
    if (bytes + offset > buffer.bytes) {
        Status err = OpenclStatus(StatusCode::kInvalidArgument,
                                  BackendErrorKind::kInvalidArgument,
                                  "Read exceeds buffer size");
        scope.SetStatus(err.code);
        return err;
    }
    cl_int err = loader_.clEnqueueReadBuffer(devices_[device_index].queue,
                                             buffer.mem, CL_TRUE, offset, bytes,
                                             data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        Status st =
            OpenclErrorStatus(StatusCode::kInternal, BackendErrorKind::kRuntime,
                              "clEnqueueReadBuffer failed", err);
        scope.SetStatus(st.code);
        return st;
    }
    return Status::OK();
}

StatusOr<OpenCLKernel> OpenCLBackend::BuildKernelFromFile(
    const std::string& path,
    const std::string& kernel_name,
    const std::string& extra_build_options) const {
    auto kernels_or =
        BuildKernelsFromFile(path, kernel_name, extra_build_options);
    if (!kernels_or.ok())
        return kernels_or.status();
    auto kernels = kernels_or.value();
    if (kernels.empty()) {
        return OpenclStatus(StatusCode::kUnavailable, BackendErrorKind::kBuild,
                            "No kernels built");
    }
    return kernels.front();
}

StatusOr<std::vector<OpenCLKernel>> OpenCLBackend::BuildKernelsFromFile(
    const std::string& path,
    const std::string& kernel_name,
    const std::string& extra_build_options) const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;
    if (devices_.empty()) {
        return OpenclStatus(StatusCode::kUnavailable,
                            BackendErrorKind::kDiscovery,
                            "No OpenCL devices available");
    }

    std::filesystem::path src_path(path);
    if (!std::filesystem::exists(src_path)) {
        auto kernel_dir = KernelDir();
        if (!kernel_dir.empty()) {
            src_path = std::filesystem::path(kernel_dir) / path;
        }
    }
    if (!std::filesystem::exists(src_path)) {
        return OpenclStatus(StatusCode::kInvalidArgument, BackendErrorKind::kIo,
                            "Kernel source not found: " + src_path.string());
    }

    std::string source;
    std::string error;
    if (!ReadFile(src_path, &source, &error)) {
        return OpenclStatus(StatusCode::kUnavailable, BackendErrorKind::kIo,
                            error);
    }

    std::vector<OpenCLKernel> out;
    std::string last_error;
    for (int i = 0; i < static_cast<int>(devices_.size()); ++i) {
        auto& dev = devices_[i];
        const std::string build_options =
            BuildOptions(dev, extra_build_options);
        const std::string cache_key =
            CacheKey(dev, kernel_name, build_options, source);

        auto program_or = BuildOrLoadProgram(dev, source, build_options,
                                             cache_key, kernel_name);
        if (!program_or.ok()) {
            last_error = program_or.status().message;
            LogBackend({LogLevel::kWarn, BackendType::kOpenCL,
                        BackendErrorKind::kBuild, "build failed: " + last_error,
                        "build", dev.desc.index, dev.desc.name});
            continue;
        }
        cl_program program = program_or.value();

        cl_int err = CL_SUCCESS;
        cl_kernel kernel =
            loader_.clCreateKernel(program, kernel_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            dev.program_cache.erase(cache_key);
            loader_.clReleaseProgram(program);
            last_error =
                "clCreateKernel failed: " + gpu::OpenCLErrorString(err);
            LogBackend({LogLevel::kWarn, BackendType::kOpenCL,
                        BackendErrorKind::kBuild, last_error, "build",
                        dev.desc.index, dev.desc.name});
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
        if (last_error.empty())
            last_error = "No OpenCL kernels built";
        return OpenclStatus(StatusCode::kUnavailable, BackendErrorKind::kBuild,
                            last_error);
    }
    return out;
}

Status OpenCLBackend::ReleaseKernel(OpenCLKernel* kernel) const {
    if (!kernel || !kernel->kernel)
        return Status::OK();
    loader_.clReleaseKernel(kernel->kernel);
    kernel->kernel = nullptr;
    kernel->program = nullptr;
    return Status::OK();
}

Status OpenCLBackend::LaunchKernel(
    const OpenCLKernel& kernel,
    const OpenCLLaunchConfig& config,
    const std::vector<OpenCLKernelArg>& args) const {
    ProfilingScope scope(profiling_.get(),
                         {ProfilingEventKind::kKernelLaunch,
                          BackendType::kOpenCL, kernel.device_index});
    scope.SetLabel(kernel.name);
    Status status = EnsureInitialized();
    if (!status.ok()) {
        scope.SetStatus(status.code);
        return status;
    }
    if (kernel.device_index < 0 ||
        kernel.device_index >= static_cast<int>(devices_.size())) {
        Status err = OpenclStatus(StatusCode::kInvalidArgument,
                                  BackendErrorKind::kInvalidArgument,
                                  "Invalid device index for kernel");
        scope.SetStatus(err.code);
        return err;
    }
    if (!kernel.kernel) {
        Status err = OpenclStatus(StatusCode::kInvalidArgument,
                                  BackendErrorKind::kInvalidArgument,
                                  "Invalid kernel handle");
        scope.SetStatus(err.code);
        return err;
    }

    cl_int err = CL_SUCCESS;
    for (cl_uint i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (arg.kind == OpenCLKernelArg::Kind::kMem) {
            cl_mem mem = arg.mem;
            err =
                loader_.clSetKernelArg(kernel.kernel, i, sizeof(cl_mem), &mem);
        } else {
            err = loader_.clSetKernelArg(kernel.kernel, i, arg.size, arg.value);
        }
        if (err != CL_SUCCESS) {
            Status st = OpenclErrorStatus(StatusCode::kInternal,
                                          BackendErrorKind::kLaunch,
                                          "clSetKernelArg failed", err);
            scope.SetStatus(st.code);
            return st;
        }
    }

    const size_t* local = config.use_local ? config.local : nullptr;
    ExecutionConfig exec_config = GetExecutionConfig();
    cl_event event = nullptr;
    cl_event* event_ptr =
        (exec_config.sync_on_launch && exec_config.kernel_timeout_ms > 0 &&
         loader_.clGetEventInfo)
            ? &event
            : nullptr;
    err = loader_.clEnqueueNDRangeKernel(
        devices_[kernel.device_index].queue, kernel.kernel, config.dims,
        nullptr, config.global, local, 0, nullptr, event_ptr);
    if (err != CL_SUCCESS) {
        Status st =
            OpenclErrorStatus(StatusCode::kInternal, BackendErrorKind::kLaunch,
                              "clEnqueueNDRangeKernel failed", err);
        scope.SetStatus(st.code);
        return st;
    }
    if (exec_config.sync_on_launch) {
        if (event && loader_.clGetEventInfo) {
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::milliseconds(exec_config.kernel_timeout_ms);
            while (true) {
                cl_int status = CL_COMPLETE;
                cl_int info_err = loader_.clGetEventInfo(
                    event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status),
                    &status, nullptr);
                if (info_err != CL_SUCCESS) {
                    if (loader_.clReleaseEvent)
                        loader_.clReleaseEvent(event);
                    Status st = OpenclErrorStatus(
                        StatusCode::kInternal, BackendErrorKind::kRuntime,
                        "clGetEventInfo failed", info_err);
                    scope.SetStatus(st.code);
                    return st;
                }
                if (status == CL_COMPLETE) {
                    break;
                }
                if (status < 0) {
                    if (loader_.clReleaseEvent)
                        loader_.clReleaseEvent(event);
                    Status st = OpenclErrorStatus(
                        StatusCode::kInternal, BackendErrorKind::kRuntime,
                        "kernel execution failed", status);
                    scope.SetStatus(st.code);
                    return st;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    if (loader_.clReleaseEvent)
                        loader_.clReleaseEvent(event);
                    Status st = OpenclStatus(
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
            if (loader_.clReleaseEvent)
                loader_.clReleaseEvent(event);
        } else {
            err = loader_.clFinish(devices_[kernel.device_index].queue);
            if (err != CL_SUCCESS) {
                Status st = OpenclErrorStatus(StatusCode::kInternal,
                                              BackendErrorKind::kRuntime,
                                              "clFinish failed", err);
                scope.SetStatus(st.code);
                return st;
            }
        }
    } else if (loader_.clFlush) {
        loader_.clFlush(devices_[kernel.device_index].queue);
    }
    return Status::OK();
}

Status OpenCLBackend::SmokeTest() const {
    Status status = EnsureInitialized();
    if (!status.ok())
        return status;

    const std::string kernel_dir = KernelDir();
    if (kernel_dir.empty()) {
        return OpenclStatus(StatusCode::kUnavailable, BackendErrorKind::kIo,
                            "OpenCL kernel directory not found");
    }

    auto kernels_or = BuildKernelsFromFile("lattice_smoke.cl", "vec_add", "");
    if (!kernels_or.ok())
        return kernels_or.status();
    auto kernels = kernels_or.value();
    if (kernels.empty()) {
        return OpenclStatus(StatusCode::kUnavailable,
                            BackendErrorKind::kDiscovery,
                            "No OpenCL devices available");
    }

    const size_t n = 1024;
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> out(n, 0.0f);

    for (const auto& kernel : kernels) {
        const int device_index = kernel.device_index;
        auto buf_a_or =
            CreateBuffer(device_index, n * sizeof(float), CL_MEM_READ_ONLY);
        if (!buf_a_or.ok())
            return buf_a_or.status();
        auto buf_b_or =
            CreateBuffer(device_index, n * sizeof(float), CL_MEM_READ_ONLY);
        if (!buf_b_or.ok())
            return buf_b_or.status();
        auto buf_out_or =
            CreateBuffer(device_index, n * sizeof(float), CL_MEM_WRITE_ONLY);
        if (!buf_out_or.ok())
            return buf_out_or.status();

        auto buf_a = buf_a_or.value();
        auto buf_b = buf_b_or.value();
        auto buf_out = buf_out_or.value();

        Status write_a =
            WriteBuffer(device_index, buf_a, a.data(), n * sizeof(float));
        if (!write_a.ok()) {
            ReleaseBuffer(&buf_a);
            ReleaseBuffer(&buf_b);
            ReleaseBuffer(&buf_out);
            return write_a;
        }
        Status write_b =
            WriteBuffer(device_index, buf_b, b.data(), n * sizeof(float));
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

        Status read =
            ReadBuffer(device_index, buf_out, out.data(), n * sizeof(float));
        if (!read.ok()) {
            ReleaseBuffer(&buf_a);
            ReleaseBuffer(&buf_b);
            ReleaseBuffer(&buf_out);
            return read;
        }

        const float expected = 3.0f;
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(out[i] - expected) > 1e-3f) {
                return OpenclStatus(
                    StatusCode::kInternal, BackendErrorKind::kRuntime,
                    "OpenCL smoke test failed: incorrect output");
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
    if (initialized_)
        return init_status_;
    initialized_ = true;

    std::string error;
    if (!loader_.Load(&error)) {
        init_status_ = OpenclStatus(StatusCode::kUnavailable,
                                    BackendErrorKind::kInit, error);
        return init_status_;
    }

    cl_uint num_platforms = 0;
    cl_int err = loader_.clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        init_status_ =
            OpenclStatus(StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
                         "No OpenCL platforms found");
        return init_status_;
    }

    num_platforms = std::min<cl_uint>(num_platforms, kMaxPlatforms);
    std::vector<cl_platform_id> platforms(num_platforms);
    err = loader_.clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        init_status_ = OpenclStatus(
            StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
            "clGetPlatformIDs failed: " + gpu::OpenCLErrorString(err));
        return init_status_;
    }

    std::vector<DeviceContext> candidates;
    std::vector<DeviceIdentity> identities;
    int device_index = 0;
    for (cl_platform_id platform : platforms) {
        cl_uint num_devices = 0;
        err = loader_.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr,
                                     &num_devices);
        if (err == CL_DEVICE_NOT_FOUND || num_devices == 0) {
            continue;
        }
        if (err != CL_SUCCESS) {
            continue;
        }
        std::vector<cl_device_id> devices(num_devices);
        err = loader_.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices,
                                     devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            continue;
        }

        const std::string platform_name =
            PlatformInfoString(platform, CL_PLATFORM_NAME);
        const std::string platform_vendor =
            PlatformInfoString(platform, CL_PLATFORM_VENDOR);
        const std::string platform_version =
            PlatformInfoString(platform, CL_PLATFORM_VERSION);

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
            ctx.desc.device_version =
                DeviceInfoString(device, CL_DEVICE_VERSION);
            ctx.desc.driver_version =
                DeviceInfoString(device, CL_DRIVER_VERSION);
            loader_.clGetDeviceInfo(device, CL_DEVICE_TYPE,
                                    sizeof(ctx.desc.type), &ctx.desc.type,
                                    nullptr);
            loader_.clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID,
                                    sizeof(ctx.desc.vendor_id),
                                    &ctx.desc.vendor_id, nullptr);
            loader_.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                    sizeof(ctx.desc.max_work_group_size),
                                    &ctx.desc.max_work_group_size, nullptr);
            loader_.clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
                                    sizeof(ctx.desc.local_mem_size),
                                    &ctx.desc.local_mem_size, nullptr);
            loader_.clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE,
                                    sizeof(ctx.desc.local_mem_type),
                                    &ctx.desc.local_mem_type, nullptr);
            loader_.clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                                    sizeof(ctx.desc.compute_units),
                                    &ctx.desc.compute_units, nullptr);
            loader_.clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                    sizeof(ctx.desc.max_clock_mhz),
                                    &ctx.desc.max_clock_mhz, nullptr);
            size_t work_item_sizes[3] = {0, 0, 0};
            loader_.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                    sizeof(work_item_sizes), work_item_sizes,
                                    nullptr);

            const std::string extensions =
                DeviceInfoString(device, CL_DEVICE_EXTENSIONS);
            ctx.caps.is_cpu = (ctx.desc.type & CL_DEVICE_TYPE_CPU) != 0;
            ctx.caps.is_gpu = (ctx.desc.type & CL_DEVICE_TYPE_GPU) != 0;
            ctx.caps.local_mem_bytes = ctx.desc.local_mem_size;
            ctx.caps.max_work_group_size = ctx.desc.max_work_group_size;
            ctx.caps.max_threads_per_block = ctx.desc.max_work_group_size;
            ctx.caps.max_work_item_sizes[0] = work_item_sizes[0];
            ctx.caps.max_work_item_sizes[1] = work_item_sizes[1];
            ctx.caps.max_work_item_sizes[2] = work_item_sizes[2];
            ctx.caps.fp64 = HasExtension(extensions, "cl_khr_fp64") ||
                                    HasExtension(extensions, "cl_amd_fp64")
                                ? CapabilityStatus::kYes
                                : CapabilityStatus::kNo;
            ctx.caps.fp16 = HasExtension(extensions, "cl_khr_fp16") ||
                                    HasExtension(extensions, "cl_amd_fp16")
                                ? CapabilityStatus::kYes
                                : CapabilityStatus::kNo;
            ctx.caps.quirks =
                QueryDeviceQuirks(BackendType::kOpenCL, ctx.desc.vendor,
                                  ctx.desc.name, ctx.desc.driver_version);
            ctx.caps.is_software =
                (ctx.caps.quirks.flags & kSoftwareEmulation) != 0;
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

    DeviceSelectionOptions selection =
        LoadDeviceSelectionOptions("LATTICE_OPENCL");
    DeviceSelectionResult selected = SelectDevices(identities, selection);
    if (selected.indices.empty()) {
        init_status_ = OpenclStatus(
            StatusCode::kUnavailable, BackendErrorKind::kDiscovery,
            selected.diagnostics.empty() ? "No OpenCL devices selected"
                                         : selected.diagnostics);
        return init_status_;
    }

    ExecutionConfig exec_config = GetExecutionConfig();
    cl_command_queue_properties queue_props =
        exec_config.enable_profiling ? CL_QUEUE_PROFILING_ENABLE : 0;

    for (int idx : selected.indices) {
        if (idx < 0 || idx >= static_cast<int>(candidates.size()))
            continue;
        DeviceContext ctx = std::move(candidates[static_cast<size_t>(idx)]);
        if (ctx.caps.quirks.disabled) {
            LogBackend({LogLevel::kWarn, BackendType::kOpenCL,
                        BackendErrorKind::kDiscovery,
                        "skipping device: " + ctx.caps.quirks.reason,
                        "device_skip", ctx.desc.index, ctx.desc.name});
            continue;
        }
        cl_int create_err = CL_SUCCESS;
        ctx.context = loader_.clCreateContext(nullptr, 1, &ctx.device, nullptr,
                                              nullptr, &create_err);
        if (create_err != CL_SUCCESS || !ctx.context) {
            LogBackend({LogLevel::kWarn, BackendType::kOpenCL,
                        BackendErrorKind::kContext, "context init failed",
                        "context", ctx.desc.index, ctx.desc.name, create_err,
                        gpu::OpenCLErrorString(create_err)});
            continue;
        }
        ctx.queue = loader_.clCreateCommandQueue(ctx.context, ctx.device,
                                                 queue_props, &create_err);
        if (create_err != CL_SUCCESS || !ctx.queue) {
            loader_.clReleaseContext(ctx.context);
            ctx.context = nullptr;
            LogBackend({LogLevel::kWarn, BackendType::kOpenCL,
                        BackendErrorKind::kContext, "queue init failed",
                        "queue", ctx.desc.index, ctx.desc.name, create_err,
                        gpu::OpenCLErrorString(create_err)});
            continue;
        }
        ctx.queue_profiling = exec_config.enable_profiling;
        devices_.push_back(std::move(ctx));
    }

    if (devices_.empty()) {
        if (!selected.indices.empty()) {
            init_status_ = OpenclStatus(
                StatusCode::kUnavailable, BackendErrorKind::kContext,
                "No OpenCL devices initialized (all selected devices failed)");
        } else {
            init_status_ = OpenclStatus(StatusCode::kUnavailable,
                                        BackendErrorKind::kDiscovery,
                                        "No OpenCL devices found");
        }
        return init_status_;
    }

    {
        DeviceMetadataStore meta_store;
        std::string meta_error;
        CacheStore& cache_store = OpenclCacheStore();
        for (const auto& dev : devices_) {
            const DeviceMetadata meta = BuildDeviceMetadata(dev.desc, dev.caps);
            std::string previous_fingerprint;
            if (!meta_store.Write(meta, &meta_error, &previous_fingerprint)) {
                LogBackend({LogLevel::kWarn, BackendType::kOpenCL,
                            BackendErrorKind::kIo,
                            "metadata persist failed: " + meta_error,
                            "metadata", dev.desc.index, dev.desc.name});
                meta_error.clear();
            } else if (!previous_fingerprint.empty() &&
                       previous_fingerprint != dev.fingerprint) {
                cache_store.InvalidateFingerprint(previous_fingerprint);
            }
        }
    }

    for (size_t i = 0; i < devices_.size(); ++i) {
        const auto& dev = devices_[i];
        std::string info = dev.desc.name + " (" +
                           DeviceTypeString(dev.desc.type) +
                           ") vendor=" + dev.desc.vendor +
                           " driver=" + dev.desc.driver_version;
        LogRecord record;
        record.level = LogLevel::kInfo;
        record.backend = BackendType::kOpenCL;
        record.kind = BackendErrorKind::kDiscovery;
        record.message = info;
        record.operation = "device_info";
        record.device_index = dev.desc.index;
        record.device_name = dev.desc.name;
        record.device_info = OpenclDeviceInfoJson(dev.desc, dev.caps);
        LogBackend(record);
    }

    init_status_ = Status::OK();
    return init_status_;
}

MemoryPool* OpenCLBackend::DevicePool(int device_index,
                                      cl_mem_flags flags) const {
    if (device_index < 0 || device_index >= static_cast<int>(devices_.size()))
        return nullptr;
    std::lock_guard<std::mutex> lock(mu_);
    auto& dev = devices_[device_index];
    auto it = dev.device_pools.find(flags);
    if (it != dev.device_pools.end())
        return it->second.get();

    MemoryPoolConfig config = OpenclDevicePoolConfig();
    const int idx = device_index;
    auto alloc_fn = [this, idx, flags](
                        size_t bytes, size_t alignment) -> StatusOr<PoolBlock> {
        (void)alignment;
        cl_int err = CL_SUCCESS;
        cl_mem buf = loader_.clCreateBuffer(devices_[idx].context, flags, bytes,
                                            nullptr, &err);
        if (err != CL_SUCCESS) {
            return OpenclStatus(
                StatusCode::kInternal, BackendErrorKind::kMemory,
                "clCreateBuffer failed: " + gpu::OpenCLErrorString(err));
        }
        PoolBlock block;
        block.key = reinterpret_cast<uintptr_t>(buf);
        block.handle = reinterpret_cast<uintptr_t>(buf);
        block.bytes = bytes;
        block.alignment = alignment;
        return block;
    };
    auto free_fn = [this](const PoolBlock& block) -> Status {
        cl_mem buf = reinterpret_cast<cl_mem>(block.handle);
        cl_int err = loader_.clReleaseMemObject(buf);
        if (err != CL_SUCCESS) {
            return OpenclStatus(
                StatusCode::kInternal, BackendErrorKind::kMemory,
                "clReleaseMemObject failed: " + gpu::OpenCLErrorString(err));
        }
        return Status::OK();
    };
    auto scrub_fn = [this, idx](const PoolBlock& block) -> Status {
        cl_mem buf = reinterpret_cast<cl_mem>(block.handle);
        return FillBufferZero(&loader_, devices_[idx].queue, buf, block.bytes);
    };

    std::ostringstream label;
    label << "opencl_device_pool_" << devices_[idx].desc.index << "_flags_"
          << flags;
    auto pool = std::make_unique<MemoryPool>(label.str(), config, alloc_fn,
                                             free_fn, scrub_fn);
    MemoryPool* out = pool.get();
    dev.device_pools.emplace(flags, std::move(pool));
    return out;
}

MemoryPool* OpenCLBackend::PinnedPool(int device_index) const {
    if (device_index < 0 || device_index >= static_cast<int>(devices_.size()))
        return nullptr;
    std::lock_guard<std::mutex> lock(mu_);
    auto& dev = devices_[device_index];
    if (dev.pinned_pool)
        return dev.pinned_pool.get();

    MemoryPoolConfig config = OpenclPinnedPoolConfig();
    const int idx = device_index;
    auto alloc_fn = [this, idx](size_t bytes,
                                size_t alignment) -> StatusOr<PoolBlock> {
        (void)alignment;
        if (!loader_.clEnqueueMapBuffer || !loader_.clEnqueueUnmapMemObject) {
            return OpenclStatus(StatusCode::kUnavailable,
                                BackendErrorKind::kMemory,
                                "OpenCL map/unmap not available");
        }
        cl_int err = CL_SUCCESS;
        cl_mem buf = loader_.clCreateBuffer(
            devices_[idx].context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            bytes, nullptr, &err);
        if (err != CL_SUCCESS) {
            return OpenclStatus(
                StatusCode::kInternal, BackendErrorKind::kMemory,
                "clCreateBuffer(host) failed: " + gpu::OpenCLErrorString(err));
        }
        void* host = loader_.clEnqueueMapBuffer(
            devices_[idx].queue, buf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
            bytes, 0, nullptr, nullptr, &err);
        if (err != CL_SUCCESS || !host) {
            loader_.clReleaseMemObject(buf);
            return OpenclStatus(
                StatusCode::kInternal, BackendErrorKind::kMemory,
                "clEnqueueMapBuffer failed: " + gpu::OpenCLErrorString(err));
        }
        PoolBlock block;
        block.key = reinterpret_cast<uintptr_t>(host);
        block.handle = reinterpret_cast<uintptr_t>(buf);
        block.host_ptr = host;
        block.bytes = bytes;
        block.alignment = alignment;
        return block;
    };
    auto free_fn = [this, idx](const PoolBlock& block) -> Status {
        if (!loader_.clEnqueueUnmapMemObject) {
            return OpenclStatus(StatusCode::kUnavailable,
                                BackendErrorKind::kMemory,
                                "OpenCL unmap not available");
        }
        cl_mem buf = reinterpret_cast<cl_mem>(block.handle);
        if (block.host_ptr) {
            cl_int unmap_err = loader_.clEnqueueUnmapMemObject(
                devices_[idx].queue, buf, block.host_ptr, 0, nullptr, nullptr);
            if (unmap_err != CL_SUCCESS) {
                return OpenclStatus(StatusCode::kInternal,
                                    BackendErrorKind::kMemory,
                                    "clEnqueueUnmapMemObject failed: " +
                                        gpu::OpenCLErrorString(unmap_err));
            }
            loader_.clFinish(devices_[idx].queue);
        }
        cl_int err = loader_.clReleaseMemObject(buf);
        if (err != CL_SUCCESS) {
            return OpenclStatus(
                StatusCode::kInternal, BackendErrorKind::kMemory,
                "clReleaseMemObject failed: " + gpu::OpenCLErrorString(err));
        }
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
    label << "opencl_pinned_pool_" << dev.desc.index;
    dev.pinned_pool = std::make_unique<MemoryPool>(label.str(), config,
                                                   alloc_fn, free_fn, scrub_fn);
    return dev.pinned_pool.get();
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
        if (!cwd.has_parent_path())
            break;
        cwd = cwd.parent_path();
    }
    return "";
}

std::string OpenCLBackend::DeviceInfoString(cl_device_id device,
                                            cl_device_info param) const {
    size_t size = 0;
    loader_.clGetDeviceInfo(device, param, 0, nullptr, &size);
    if (size == 0)
        return "";
    std::string out(size, '\0');
    loader_.clGetDeviceInfo(device, param, size, out.data(), nullptr);
    if (!out.empty() && out.back() == '\0')
        out.pop_back();
    return out;
}

std::string OpenCLBackend::PlatformInfoString(cl_platform_id platform,
                                              cl_platform_info param) const {
    size_t size = 0;
    loader_.clGetPlatformInfo(platform, param, 0, nullptr, &size);
    if (size == 0)
        return "";
    std::string out(size, '\0');
    loader_.clGetPlatformInfo(platform, param, size, out.data(), nullptr);
    if (!out.empty() && out.back() == '\0')
        out.pop_back();
    return out;
}

std::string OpenCLBackend::BuildOptions(const DeviceContext& dev,
                                        const std::string& extra) const {
    std::string options;
    std::string c_version =
        DeviceInfoString(dev.device, CL_DEVICE_OPENCL_C_VERSION);
    std::string std_opt = OpenclCStd(c_version);
    AppendOption(&options, std_opt);
    const std::string kernel_dir = KernelDir();
    AppendOption(&options, BuildIncludeOption(kernel_dir, "-I"));

    KernelBuildDefines defs;
    defs.backend = BackendType::kOpenCL;
    defs.device_index = dev.desc.index;
    defs.abi_version = opencl::kAbiVersion;
    defs.abi_version_min = opencl::kAbiVersionMin;
    defs.device_type = static_cast<uint64_t>(dev.desc.type);
    defs.vendor_id = static_cast<uint64_t>(dev.desc.vendor_id);
    defs.fast_math = exec_config_.enable_fast_math;
    defs.vectorize = exec_config_.enable_vectorize;
    defs.mixed_precision = exec_config_.enable_mixed_precision;
    std::string extensions = DeviceInfoString(dev.device, CL_DEVICE_EXTENSIONS);
    if (extensions.find("cl_khr_fp64") != std::string::npos) {
        defs.has_fp64 = true;
    }
    if (extensions.find("cl_khr_fp16") != std::string::npos) {
        defs.has_fp16 = true;
    }
    std::string defines = KernelDefineString(defs, "-D");
    AppendOption(&options, defines);

    std::string env_opts = LoadBuildOptionsEnv("LATTICE_OPENCL_BUILD_OPTIONS");
    AppendOption(&options, env_opts);
    AppendOption(&options, extra);
    if (exec_config_.enable_fast_math) {
        AppendOption(&options, "-cl-fast-relaxed-math");
        AppendOption(&options, "-cl-mad-enable");
    }
    if (KernelDebugEnabled("LATTICE_OPENCL_BUILD_DEBUG")) {
        AppendOption(&options, KernelDebugOptions(BackendType::kOpenCL));
    }

    LogBackend({LogLevel::kDebug, BackendType::kOpenCL,
                BackendErrorKind::kBuild, "build_options " + options,
                "build_options", dev.desc.index, dev.desc.name});

    return options;
}

std::string OpenCLBackend::CacheKey(const DeviceContext& dev,
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
    return "opencl_" + meta_hash + "_" + src_hash;
}

StatusOr<cl_program> OpenCLBackend::BuildOrLoadProgram(
    DeviceContext& dev,
    const std::string& source,
    const std::string& build_options,
    const std::string& cache_key,
    const std::string& kernel_name) const {
    auto it = dev.program_cache.find(cache_key);
    if (it != dev.program_cache.end()) {
        return it->second;
    }
    CacheStore& store = OpenclCacheStore();
    ::lattice::runtime::CacheKey store_key{cache_key, dev.fingerprint};
    const bool rebuild_on_failure =
        KernelRebuildOnFailureEnabled("LATTICE_OPENCL_REBUILD_ON_FAILURE");

    KernelTrace trace;
    trace.backend = BackendType::kOpenCL;
    trace.kernel_name = kernel_name;
    trace.build_options = build_options;
    trace.source = source;
    trace.device_index = dev.desc.index;
    trace.device_name = dev.desc.name;
    trace.enabled = exec_config_.trace_kernels;
    std::string trace_path;
    if (TraceKernelSource(trace, &trace_path)) {
        LogBackend({LogLevel::kTrace, BackendType::kOpenCL,
                    BackendErrorKind::kBuild, "kernel trace written", "build",
                    dev.desc.index, dev.desc.name, 0, "", trace_path});
    }
    std::string binary;
    std::string error;
    if (store.ReadBinary(store_key, &binary, &error)) {
        const unsigned char* bin_ptr =
            reinterpret_cast<const unsigned char*>(binary.data());
        size_t bin_size = binary.size();
        cl_int status = CL_SUCCESS;
        cl_program program = loader_.clCreateProgramWithBinary(
            dev.context, 1, &dev.device, &bin_size, &bin_ptr, nullptr, &status);
        if (status == CL_SUCCESS && program) {
            cl_int err =
                loader_.clBuildProgram(program, 1, &dev.device,
                                       build_options.c_str(), nullptr, nullptr);
            if (err == CL_SUCCESS) {
                dev.program_cache[cache_key] = program;
                return program;
            }
            std::string log = ProgramBuildLog(loader_, program, dev.device);
            loader_.clReleaseProgram(program);
            if (!rebuild_on_failure) {
                std::string message = "cached OpenCL binary build failed";
                if (!log.empty())
                    message += ": " + log;
                return OpenclStatus(StatusCode::kInternal,
                                    BackendErrorKind::kBuild, message);
            }
        }
        store.Invalidate(store_key);
        if (!rebuild_on_failure) {
            return OpenclStatus(StatusCode::kInternal,
                                BackendErrorKind::kCompile,
                                "cached OpenCL binary load failed: " +
                                    gpu::OpenCLErrorString(status));
        }
    }

    const char* src = source.c_str();
    cl_int err = CL_SUCCESS;
    cl_program program =
        loader_.clCreateProgramWithSource(dev.context, 1, &src, nullptr, &err);
    if (err != CL_SUCCESS || !program) {
        return OpenclStatus(
            StatusCode::kInternal, BackendErrorKind::kCompile,
            "clCreateProgramWithSource failed: " + gpu::OpenCLErrorString(err));
    }

    err = loader_.clBuildProgram(program, 1, &dev.device, build_options.c_str(),
                                 nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::string log = ProgramBuildLog(loader_, program, dev.device);
        LogBackend({LogLevel::kWarn, BackendType::kOpenCL,
                    BackendErrorKind::kBuild, "OpenCL build failed", "build",
                    dev.desc.index, dev.desc.name, err,
                    gpu::OpenCLErrorString(err), "", log});
        loader_.clReleaseProgram(program);
        return OpenclStatus(StatusCode::kInternal, BackendErrorKind::kBuild,
                            "OpenCL build failed: " + log);
    }

    size_t bin_size = 0;
    err = loader_.clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                                   sizeof(size_t), &bin_size, nullptr);
    if (err == CL_SUCCESS && bin_size > 0) {
        std::vector<unsigned char> bin(bin_size);
        unsigned char* bin_ptr = bin.data();
        err =
            loader_.clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                                     sizeof(unsigned char*), &bin_ptr, nullptr);
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
