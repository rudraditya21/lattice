#include "runtime/backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "runtime/backends/backend_log.h"
#include "runtime/backends/memory_pool.h"
#include "runtime/backends/memory_stats.h"
#include "runtime/backends/memory_utils.h"

#ifdef __linux__
#include <numaif.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace lattice::runtime {

namespace {

constexpr size_t kDefaultAlignment = 64;
constexpr uint64_t kCanary = 0xDEADBEEFCAFEBABEULL;
constexpr size_t kCanaryBytes = sizeof(uint64_t);

struct CpuPoolState {
    std::mutex mu;
    std::unique_ptr<MemoryPool> device_pool;
    std::unique_ptr<MemoryPool> pinned_pool;
};

void WriteCanary(void* ptr) {
    uint64_t value = kCanary;
    std::memcpy(ptr, &value, sizeof(value));
}

bool CheckCanary(const void* ptr) {
    uint64_t value = 0;
    std::memcpy(&value, ptr, sizeof(value));
    return value == kCanary;
}

MemoryPoolConfig CpuDevicePoolConfig() {
    static MemoryPoolConfig config = [] {
        MemoryPoolConfig base = DefaultDevicePoolConfig();
        base.scrub_on_free = true;
        base.zero_on_alloc = true;
        base.bucket_bytes = kDefaultAlignment;
        base = LoadMemoryPoolConfig("LATTICE_DEVICE_POOL", base);
        base = LoadMemoryPoolConfig("LATTICE_CPU_DEVICE_POOL", base);
        return base;
    }();
    return config;
}

MemoryPoolConfig CpuPinnedPoolConfig() {
    static MemoryPoolConfig config = [] {
        MemoryPoolConfig base = DefaultPinnedPoolConfig();
        base.scrub_on_free = true;
        base.zero_on_alloc = true;
        base.bucket_bytes = kDefaultAlignment;
        base = LoadMemoryPoolConfig("LATTICE_PINNED_POOL", base);
        base = LoadMemoryPoolConfig("LATTICE_CPU_PINNED_POOL", base);
        return base;
    }();
    return config;
}

StatusOr<PoolBlock> CpuPoolAlloc(size_t bytes, size_t alignment) {
    PoolBlock block;
    if (bytes == 0)
        return block;
    if (alignment == 0)
        alignment = kDefaultAlignment;
    size_t total = bytes + 2 * kCanaryBytes;
    size_t alloc_align = std::max(alignment, kDefaultAlignment);
    void* raw = nullptr;
#if defined(_MSC_VER)
    raw = _aligned_malloc(total, alloc_align);
    if (!raw) {
        return Status::Internal("cpu alloc failed");
    }
#else
    if (posix_memalign(&raw, alloc_align, total) != 0 || !raw) {
        return Status::Internal("cpu alloc failed");
    }
#endif
    void* user_ptr = static_cast<char*>(raw) + kCanaryBytes;
    WriteCanary(raw);
    WriteCanary(static_cast<char*>(user_ptr) + bytes);
    block.key = reinterpret_cast<uintptr_t>(user_ptr);
    block.handle = reinterpret_cast<uintptr_t>(raw);
    block.host_ptr = user_ptr;
    block.bytes = bytes;
    block.alignment = alignment;
    return block;
}

Status CpuPoolFree(const PoolBlock& block) {
    void* raw = reinterpret_cast<void*>(block.handle);
    if (!raw)
        return Status::OK();
#if defined(_MSC_VER)
    _aligned_free(raw);
#else
    free(raw);
#endif
    return Status::OK();
}

Status CpuPoolScrub(const PoolBlock& block, bool secure) {
    if (block.handle == 0)
        return Status::OK();
    size_t requested =
        block.requested_bytes ? block.requested_bytes : block.bytes;
    if (requested == 0)
        return Status::OK();
    if (requested > block.bytes)
        requested = block.bytes;
    auto* raw = reinterpret_cast<unsigned char*>(block.handle);
    auto* user = block.host_ptr
                     ? static_cast<unsigned char*>(block.host_ptr)
                     : raw + static_cast<std::ptrdiff_t>(kCanaryBytes);
    if (!block.scrub_on_alloc) {
        if (!CheckCanary(raw) || !CheckCanary(user + requested)) {
            return Status::Internal("memory canary corrupted");
        }
    }
    if (secure) {
        SecureZero(user, requested);
    } else {
        std::memset(user, 0, requested);
    }
    WriteCanary(raw);
    WriteCanary(user + requested);
    return Status::OK();
}

CpuPoolState& CpuPools() {
    static CpuPoolState state;
    return state;
}

MemoryPool* CpuDevicePool() {
    auto& state = CpuPools();
    std::lock_guard<std::mutex> lock(state.mu);
    if (!state.device_pool) {
        MemoryPoolConfig config = CpuDevicePoolConfig();
        auto scrub_fn = [secure = config.secure_scrub](const PoolBlock& block) {
            return CpuPoolScrub(block, secure);
        };
        state.device_pool = std::make_unique<MemoryPool>(
            "cpu_device_pool", config, CpuPoolAlloc, CpuPoolFree, scrub_fn);
    }
    return state.device_pool.get();
}

MemoryPool* CpuPinnedPool() {
    auto& state = CpuPools();
    std::lock_guard<std::mutex> lock(state.mu);
    if (!state.pinned_pool) {
        MemoryPoolConfig config = CpuPinnedPoolConfig();
        auto scrub_fn = [secure = config.secure_scrub](const PoolBlock& block) {
            return CpuPoolScrub(block, secure);
        };
        state.pinned_pool = std::make_unique<MemoryPool>(
            "cpu_pinned_pool", config, CpuPoolAlloc, CpuPoolFree, scrub_fn);
    }
    return state.pinned_pool.get();
}

StatusOr<Allocation> CpuAllocateFromPool(MemoryPool* pool,
                                         AllocationKind kind,
                                         size_t bytes,
                                         size_t alignment,
                                         int numa_node) {
    if (!pool) {
        return Status::Internal("cpu pool unavailable");
    }
    if (alignment == 0)
        alignment = kDefaultAlignment;
    Allocation alloc;
    alloc.bytes = bytes;
    alloc.alignment = alignment;
    alloc.numa_node = numa_node;
    alloc.kind = kind;
    auto block_or = pool->Acquire(bytes, alignment);
    if (!block_or.ok())
        return block_or.status();
    PoolBlock block = block_or.value();
    void* raw = reinterpret_cast<void*>(block.handle);
    void* user_ptr =
        block.host_ptr
            ? block.host_ptr
            : (raw ? static_cast<char*>(raw) + kCanaryBytes : nullptr);
    if (user_ptr && bytes > 0 && raw) {
        WriteCanary(raw);
        WriteCanary(static_cast<char*>(user_ptr) + bytes);
    }
    alloc.ptr = user_ptr;
    alloc.device_handle = reinterpret_cast<void*>(
        block.device_ptr ? block.device_ptr : block.handle);
    alloc.from_pool = block.from_pool;
#ifdef __linux__
    if (numa_node >= 0 && raw && block.bytes > 0) {
        unsigned long nodemask = 1UL << numa_node;
        size_t total = block.bytes + 2 * kCanaryBytes;
        long mbind_res =
            syscall(SYS_mbind, raw, total, MPOL_PREFERRED, &nodemask,
                    sizeof(nodemask) * 8, MPOL_MF_STRICT);
        if (mbind_res != 0) {
            return Status::Internal("numa mbind failed");
        }
    }
#endif
    return alloc;
}

std::string NormalizeBackendName(const char* name) {
    std::string out;
    if (!name)
        return out;
    for (const char* p = name; *p; ++p) {
        out.push_back(static_cast<char>(std::tolower(*p)));
    }
    return out;
}

bool BackendAvailable(const Backend* backend) {
    if (!backend)
        return false;
    auto stream_or = backend->CreateStream();
    return stream_or.ok();
}

const Backend* SelectBestAvailableBackend() {
    const Backend* candidates[] = {
        GetCudaBackend(),   GetHipBackend(),
#if defined(__APPLE__)
        GetMetalBackend(),
#endif
        GetOpenCLBackend(), GetCpuBackend(),
    };
    for (const auto* candidate : candidates) {
        if (BackendAvailable(candidate))
            return candidate;
    }
    return GetCpuBackend();
}

std::string BackendNameForEnv(BackendType type) {
    switch (type) {
        case BackendType::kCPU:
            return "cpu";
        case BackendType::kOpenCL:
            return "opencl";
        case BackendType::kCUDA:
            return "cuda";
        case BackendType::kHIP:
            return "hip";
        case BackendType::kMetal:
            return "metal";
    }
    return "cpu";
}

class ThreadPool {
   public:
    explicit ThreadPool(int threads)
        : shutdown_(false), deterministic_(false), next_queue_(0) {
        if (threads <= 0)
            threads = std::max(
                1, static_cast<int>(std::thread::hardware_concurrency()));
        queues_.resize(static_cast<size_t>(threads));
        for (int i = 0; i < threads; ++i) {
            workers_.emplace_back([this]() { this->WorkerLoop(); });
        }
    }
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(mu_);
            shutdown_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable())
                t.join();
        }
    }

    std::future<void> Submit(std::function<void()> fn) {
        auto task = std::make_shared<std::packaged_task<void()>>(std::move(fn));
        std::future<void> fut = task->get_future();
        size_t idx = next_queue_.fetch_add(1, std::memory_order_relaxed) %
                     queues_.size();
        {
            std::unique_lock<std::mutex> lock(mu_);
            queues_[idx].push_back(
                {priority_counter_++, [task]() { (*task)(); }});
        }
        cv_.notify_one();
        return fut;
    }

    static ThreadPool& Instance() {
        static ThreadPool pool(
            std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
        return pool;
    }

    void SetDeterministic(bool deterministic) {
        std::unique_lock<std::mutex> lock(mu_);
        deterministic_ = deterministic;
    }

   private:
    void WorkerLoop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] { return shutdown_ || HasWorkLocked(); });
                if (shutdown_ && !HasWorkLocked())
                    return;
                job = PopTaskLocked();
            }
            // Simple task fusion: if more tasks available locally, pull up to 3
            // and run inline.
            if (!deterministic_) {
                std::vector<std::function<void()>> batch;
                {
                    std::unique_lock<std::mutex> lock(mu_);
                    for (int i = 0; i < 3 && HasWorkLocked(); ++i) {
                        batch.push_back(PopTaskLocked());
                    }
                }
                job();
                for (auto& t : batch)
                    t();
            } else {
                job();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::vector<std::deque<std::pair<uint64_t, std::function<void()>>>> queues_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool shutdown_;
    bool deterministic_;
    uint64_t priority_counter_ = 0;
    std::atomic<size_t> next_queue_;

    bool HasWorkLocked() const {
        for (const auto& q : queues_) {
            if (!q.empty())
                return true;
        }
        return false;
    }

    std::function<void()> PopTaskLocked() {
        // Prefer own queue (round-robin).
        size_t start =
            next_queue_.load(std::memory_order_relaxed) % queues_.size();
        // deterministic: pop front; else use priority (largest key) and steal.
        if (deterministic_) {
            for (size_t i = 0; i < queues_.size(); ++i) {
                size_t idx = (start + i) % queues_.size();
                if (!queues_[idx].empty()) {
                    auto task = queues_[idx].front();
                    queues_[idx].pop_front();
                    return task.second;
                }
            }
        } else {
            // Non-deterministic: steal highest priority available.
            uint64_t best_pri = 0;
            size_t best_idx = queues_.size();
            bool found = false;
            for (size_t i = 0; i < queues_.size(); ++i) {
                if (!queues_[i].empty()) {
                    auto pri = queues_[i].back().first;
                    if (!found || pri > best_pri) {
                        best_pri = pri;
                        best_idx = i;
                        found = true;
                    }
                }
            }
            if (found) {
                auto task = queues_[best_idx].back();
                queues_[best_idx].pop_back();
                return task.second;
            }
        }
        return []() {};
    }
};

class CpuEvent final : public Event {
   public:
    void Record() override {
        timestamp_ = std::chrono::steady_clock::now();
        has_timestamp_ = true;
        ready_ = std::make_shared<std::promise<void>>();
        future_ = ready_->get_future();
        ready_->set_value();
    }
    void Wait() override {
        if (future_.valid())
            future_.wait();
    }
    bool Ready() const override {
        return future_.valid() && future_.wait_for(std::chrono::seconds(0)) ==
                                      std::future_status::ready;
    }

    bool HasTimestamp() const { return has_timestamp_; }
    std::chrono::steady_clock::time_point Timestamp() const {
        return timestamp_;
    }

   private:
    std::shared_ptr<std::promise<void>> ready_;
    std::future<void> future_;
    std::chrono::steady_clock::time_point timestamp_{};
    bool has_timestamp_ = false;
};

class CpuStream final : public Stream {
   public:
    void Submit(std::function<void()> fn) override {
        std::vector<std::shared_ptr<Event>> deps;
        std::shared_future<void> prev;
        {
            std::lock_guard<std::mutex> lock(mu_);
            deps.swap(deps_);
            prev = tail_;
        }
        for (auto& dep : deps) {
            dep->Wait();
        }
        if (prev.valid()) {
            prev.wait();
        }
        // Tag task with stream priority to guide scheduler.
        std::shared_future<void> fut =
            ThreadPool::Instance().Submit(std::move(fn)).share();
        {
            std::lock_guard<std::mutex> lock(mu_);
            tail_ = fut;
        }
    }
    void Synchronize() override {
        std::shared_future<void> tail;
        {
            std::lock_guard<std::mutex> lock(mu_);
            tail = tail_;
            deps_.clear();
        }
        if (tail.valid()) {
            tail.wait();
        }
    }
    void AddDependency(const std::shared_ptr<Event>& ev) override {
        if (!ev)
            return;
        std::lock_guard<std::mutex> lock(mu_);
        deps_.push_back(ev);
    }
    StatusOr<std::shared_ptr<Event>> CreateEvent() const override {
        return std::make_shared<CpuEvent>();
    }
    void RecordEvent(const std::shared_ptr<Event>& ev) override {
        if (!ev)
            return;
        Submit([ev]() { ev->Record(); });
    }
    void SetPriority(int priority) override { priority_ = priority; }

   private:
    mutable std::mutex mu_;
    std::vector<std::shared_ptr<Event>> deps_;
    std::shared_future<void> tail_;
    int priority_ = 0;
};

BackendCapabilities CpuCaps() {
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
    caps.supported_dtypes = {DType::kBool,    DType::kI8,       DType::kI16,
                             DType::kI32,     DType::kI64,      DType::kU8,
                             DType::kU16,     DType::kU32,      DType::kU64,
                             DType::kF16,     DType::kBF16,     DType::kF32,
                             DType::kF64,     DType::kC64,      DType::kC128,
                             DType::kDecimal, DType::kRational, DType::kTensor};
    return caps;
}

}  // namespace

CpuBackend::CpuBackend() {
    const char* env = std::getenv("LATTICE_NUMA_NODE");
    if (env) {
        try {
            preferred_numa_node_ = std::stoi(env);
        } catch (...) {
            preferred_numa_node_ = -1;
        }
    }
    default_priority_ = 0;
}

CpuBackend::~CpuBackend() {
    auto& state = CpuPools();
    std::lock_guard<std::mutex> lock(state.mu);
    if (state.device_pool) {
        MemoryPoolStats stats = state.device_pool->Stats();
        if (HasOutstandingAllocs(stats)) {
            LogBackend({LogLevel::kWarn, BackendType::kCPU,
                        BackendErrorKind::kMemory,
                        "device pool has outstanding allocations (" +
                            FormatPoolStats(stats) + ")",
                        "device_pool_leak", 0, "cpu"});
        }
        state.device_pool->Trim();
    }
    if (state.pinned_pool) {
        MemoryPoolStats stats = state.pinned_pool->Stats();
        if (HasOutstandingAllocs(stats)) {
            LogBackend({LogLevel::kWarn, BackendType::kCPU,
                        BackendErrorKind::kMemory,
                        "pinned pool has outstanding allocations (" +
                            FormatPoolStats(stats) + ")",
                        "pinned_pool_leak", -1, "cpu"});
        }
        state.pinned_pool->Trim();
    }
}

BackendType CpuBackend::Type() const {
    return BackendType::kCPU;
}

std::string CpuBackend::Name() const {
    return "cpu";
}

BackendCapabilities CpuBackend::Capabilities() const {
    return CpuCaps();
}

StatusOr<std::shared_ptr<Stream>> CpuBackend::CreateStream() const {
    return std::make_shared<CpuStream>();
}

StatusOr<std::shared_ptr<Event>> CpuBackend::CreateEvent() const {
    return std::make_shared<CpuEvent>();
}

StatusOr<Allocation> CpuBackend::Allocate(size_t bytes,
                                          size_t alignment) const {
    return CpuAllocateFromPool(CpuDevicePool(), AllocationKind::kHost, bytes,
                               alignment, preferred_numa_node_);
}

Status CpuBackend::Deallocate(const Allocation& alloc) const {
    if (!alloc.ptr)
        return Status::OK();
    MemoryPool* pool = CpuDevicePool();
    if (!pool) {
        return Status::Internal("cpu device pool unavailable");
    }
    return pool->Release(reinterpret_cast<uintptr_t>(alloc.ptr));
}

StatusOr<Allocation> CpuBackend::AllocatePinned(size_t bytes,
                                                size_t alignment) const {
    return CpuAllocateFromPool(CpuPinnedPool(), AllocationKind::kPinnedHost,
                               bytes, alignment, preferred_numa_node_);
}

Status CpuBackend::DeallocatePinned(const Allocation& alloc) const {
    if (!alloc.ptr)
        return Status::OK();
    MemoryPool* pool = CpuPinnedPool();
    if (!pool) {
        return Status::Internal("cpu pinned pool unavailable");
    }
    return pool->Release(reinterpret_cast<uintptr_t>(alloc.ptr));
}

int CpuBackend::NumThreads() const {
    return static_cast<int>(std::thread::hardware_concurrency());
}

size_t CpuBackend::OutstandingAllocs() const {
    size_t total = 0;
    if (auto* pool = CpuDevicePool()) {
        total += pool->Outstanding();
    }
    if (auto* pool = CpuPinnedPool()) {
        total += pool->Outstanding();
    }
    return total;
}

BackendMemoryStats CpuBackend::MemoryStats() const {
    BackendMemoryStats stats;
    if (auto* pool = CpuDevicePool()) {
        stats.device = pool->Stats();
    }
    if (auto* pool = CpuPinnedPool()) {
        stats.pinned = pool->Stats();
    }
    return stats;
}

std::vector<DeviceMemoryStats> CpuBackend::MemoryStatsByDevice() const {
    DeviceMemoryStats entry;
    entry.backend = BackendType::kCPU;
    entry.device_index = 0;
    entry.device_name = "cpu";
    if (auto* pool = CpuDevicePool()) {
        entry.device = pool->Stats();
    }
    if (auto* pool = CpuPinnedPool()) {
        entry.pinned = pool->Stats();
    }
    return {entry};
}

ExecutionConfig CpuBackend::GetExecutionConfig() const {
    std::lock_guard<std::mutex> lock(config_mu_);
    return exec_config_;
}

Status CpuBackend::SetExecutionConfig(const ExecutionConfig& config) {
    std::lock_guard<std::mutex> lock(config_mu_);
    exec_config_ = config;
    return Status::OK();
}

StatusOr<uint64_t> CpuBackend::ElapsedNs(
    const std::shared_ptr<Event>& start,
    const std::shared_ptr<Event>& end) const {
    if (!start || !end) {
        return Status::Invalid("Missing events for elapsed time");
    }
    ExecutionConfig config = GetExecutionConfig();
    if (!config.enable_profiling) {
        return Status::Unavailable("CPU profiling disabled");
    }
    auto* start_ev = dynamic_cast<CpuEvent*>(start.get());
    auto* end_ev = dynamic_cast<CpuEvent*>(end.get());
    if (!start_ev || !end_ev) {
        return Status::Invalid("Events do not belong to CPU backend");
    }
    if (!start_ev->HasTimestamp() || !end_ev->HasTimestamp()) {
        return Status::Invalid("Events have not been recorded");
    }
    auto delta = end_ev->Timestamp() - start_ev->Timestamp();
    auto ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(delta).count();
    if (ns < 0)
        ns = 0;
    return static_cast<uint64_t>(ns);
}

void CpuBackend::SetDefaultPriority(int priority) {
    default_priority_ = priority;
}

void CpuBackend::SetDeterministic(bool deterministic) {
    ThreadPool::Instance().SetDeterministic(deterministic);
}

const Backend* GetCpuBackend() {
    static CpuBackend* backend = [] { return new CpuBackend(); }();
    return backend;
}

const Backend* GetBackendByType(BackendType type) {
    switch (type) {
        case BackendType::kCPU:
            return GetCpuBackend();
        case BackendType::kOpenCL:
            return GetOpenCLBackend();
        case BackendType::kCUDA:
            return GetCudaBackend();
        case BackendType::kHIP:
            return GetHipBackend();
        case BackendType::kMetal:
#if defined(__APPLE__)
            return GetMetalBackend();
#else
            return nullptr;
#endif
    }
    return GetCpuBackend();
}

const Backend* GetDefaultBackend() {
    const char* env = std::getenv("LATTICE_BACKEND");
    const std::string name = NormalizeBackendName(env);
    if (!name.empty()) {
        BackendType requested = BackendType::kCPU;
        if (name == "opencl" || name == "ocl") {
            requested = BackendType::kOpenCL;
        } else if (name == "cuda") {
            requested = BackendType::kCUDA;
        } else if (name == "hip") {
            requested = BackendType::kHIP;
        } else if (name == "metal" || name == "mtl") {
            requested = BackendType::kMetal;
        } else if (name == "cpu") {
            requested = BackendType::kCPU;
        } else if (name == "auto") {
            return SelectBestAvailableBackend();
        } else {
            const Backend* fallback = SelectBestAvailableBackend();
            std::string message = "Unknown backend '" + name +
                                  "', falling back to '" +
                                  BackendNameForEnv(fallback->Type()) + "'";
            std::cerr << message << "\n";
            LogBackend({LogLevel::kWarn, BackendType::kCPU,
                        BackendErrorKind::kInvalidArgument, message,
                        "backend_select"});
            return fallback;
        }
        const Backend* backend = GetBackendByType(requested);
        if (BackendAvailable(backend)) {
            return backend;
        }
        const Backend* fallback = SelectBestAvailableBackend();
        std::string message = "Requested backend '" + name +
                              "' is unavailable; falling back to '" +
                              BackendNameForEnv(fallback->Type()) + "'";
        std::cerr << message << "\n";
        LogBackend({LogLevel::kWarn, BackendType::kCPU,
                    BackendErrorKind::kDiscovery, message, "backend_select"});
        return fallback;
    }
    return GetCpuBackend();
}

}  // namespace lattice::runtime
