#include "runtime/backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <new>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "runtime/backends/backend_log.h"

#ifdef __linux__
#include <numaif.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace lattice::runtime {

namespace {

constexpr size_t kDefaultAlignment = 64;
constexpr uint64_t kCanary = 0xDEADBEEFCAFEBABEULL;
constexpr size_t kPoolMaxSize = 64 * 1024;

struct AllocInfo {
  size_t bytes;
  size_t alignment;
  uint64_t* pre_guard;
  uint64_t* post_guard;
  void* user_ptr;
  void* raw_ptr;
  bool from_pool;
};

std::mutex g_alloc_mu;
std::unordered_map<void*, AllocInfo> g_allocs;
std::unordered_map<size_t, std::vector<void*>> g_pool;
MemoryPoolStats g_pool_stats;

std::string NormalizeBackendName(const char* name) {
  std::string out;
  if (!name) return out;
  for (const char* p = name; *p; ++p) {
    out.push_back(static_cast<char>(std::tolower(*p)));
  }
  return out;
}

bool BackendAvailable(const Backend* backend) {
  if (!backend) return false;
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
    if (BackendAvailable(candidate)) return candidate;
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
  explicit ThreadPool(int threads) : shutdown_(false), deterministic_(false), next_queue_(0) {
    if (threads <= 0) threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
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
      if (t.joinable()) t.join();
    }
  }

  std::future<void> Submit(std::function<void()> fn) {
    auto task = std::make_shared<std::packaged_task<void()>>(std::move(fn));
    std::future<void> fut = task->get_future();
    size_t idx = next_queue_.fetch_add(1, std::memory_order_relaxed) % queues_.size();
    {
      std::unique_lock<std::mutex> lock(mu_);
      queues_[idx].push_back({priority_counter_++, [task]() { (*task)(); }});
    }
    cv_.notify_one();
    return fut;
  }

  static ThreadPool& Instance() {
    static ThreadPool pool(std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
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
        if (shutdown_ && !HasWorkLocked()) return;
        job = PopTaskLocked();
      }
      // Simple task fusion: if more tasks available locally, pull up to 3 and run inline.
      if (!deterministic_) {
        std::vector<std::function<void()>> batch;
        {
          std::unique_lock<std::mutex> lock(mu_);
          for (int i = 0; i < 3 && HasWorkLocked(); ++i) {
            batch.push_back(PopTaskLocked());
          }
        }
        job();
        for (auto& t : batch) t();
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
      if (!q.empty()) return true;
    }
    return false;
  }

  std::function<void()> PopTaskLocked() {
    // Prefer own queue (round-robin).
    size_t start = next_queue_.load(std::memory_order_relaxed) % queues_.size();
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

class CpuStream final : public Stream {
 public:
  void Submit(std::function<void()> fn) override {
    for (auto& dep : deps_) {
      dep->Wait();
    }
    // Tag task with stream priority to guide scheduler.
    futures_.emplace_back(ThreadPool::Instance().Submit(std::move(fn)));
  }
  void Synchronize() override {
    for (auto& f : futures_) f.get();
    futures_.clear();
    deps_.clear();
  }
  void AddDependency(const std::shared_ptr<Event>& ev) override { deps_.push_back(ev); }
  void SetPriority(int priority) override { priority_ = priority; }

 private:
  std::vector<std::future<void>> futures_;
  std::vector<std::shared_ptr<Event>> deps_;
  int priority_ = 0;
};

class CpuEvent final : public Event {
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
  caps.supported_dtypes = {
      DType::kBool, DType::kI8,  DType::kI16,  DType::kI32,     DType::kI64,      DType::kU8,
      DType::kU16,  DType::kU32, DType::kU64,  DType::kF16,     DType::kBF16,     DType::kF32,
      DType::kF64,  DType::kC64, DType::kC128, DType::kDecimal, DType::kRational, DType::kTensor};
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

StatusOr<Allocation> CpuBackend::Allocate(size_t bytes, size_t alignment) const {
  Allocation alloc;
  alloc.bytes = bytes;
  alloc.alignment = alignment;
  alloc.numa_node = preferred_numa_node_;
  alloc.kind = AllocationKind::kHost;
  const bool use_pool = bytes > 0 && bytes <= kPoolMaxSize && alignment <= kDefaultAlignment;
  void* pooled = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_alloc_mu);
    g_pool_stats.total_alloc_calls++;
    if (use_pool) {
      auto it = g_pool.find(bytes);
      if (it != g_pool.end() && !it->second.empty()) {
        pooled = it->second.back();
        it->second.pop_back();
        g_pool_stats.pool_hits++;
        if (g_pool_stats.cached_blocks > 0) {
          g_pool_stats.cached_blocks--;
        }
        if (g_pool_stats.cached_bytes >= bytes) {
          g_pool_stats.cached_bytes -= bytes;
        } else {
          g_pool_stats.cached_bytes = 0;
        }
      }
    }
    if (use_pool && !pooled) {
      g_pool_stats.pool_misses++;
    }
  }

  void* raw = pooled;
  size_t canary_bytes = sizeof(uint64_t);
  size_t total = bytes + 2 * canary_bytes;
  size_t alloc_align = std::max(alignment, kDefaultAlignment);

  if (!raw) {
#if defined(_MSC_VER)
    raw = _aligned_malloc(total, alloc_align);
    if (!raw) return Status::Internal("cpu alloc failed");
#else
    if (posix_memalign(&raw, alloc_align, total) != 0) {
      return Status::Internal("cpu alloc failed");
    }
#endif
  }

  void* user_ptr = static_cast<void*>(static_cast<char*>(raw) + canary_bytes);
  auto* pre = reinterpret_cast<uint64_t*>(raw);
  auto* post = reinterpret_cast<uint64_t*>(static_cast<char*>(user_ptr) + bytes);
  *pre = kCanary;
  *post = kCanary;
  if (user_ptr && bytes > 0) {
    std::memset(user_ptr, 0, bytes);  // zero-init for safety
  }
  alloc.ptr = user_ptr;
  alloc.from_pool = use_pool && pooled != nullptr;

  {
    std::lock_guard<std::mutex> lock(g_alloc_mu);
    g_allocs[user_ptr] = {bytes, alignment, pre, post, user_ptr, raw, alloc.from_pool};
    g_pool_stats.in_use_bytes += bytes;
    g_pool_stats.in_use_blocks++;
    g_pool_stats.peak_in_use_bytes =
        std::max(g_pool_stats.peak_in_use_bytes, g_pool_stats.in_use_bytes);
    g_pool_stats.peak_in_use_blocks =
        std::max(g_pool_stats.peak_in_use_blocks, g_pool_stats.in_use_blocks);
  }
#ifdef __linux__
  if (preferred_numa_node_ >= 0) {
    unsigned long nodemask = 1UL << preferred_numa_node_;
    long mbind_res = syscall(SYS_mbind, raw, total, MPOL_PREFERRED, &nodemask, sizeof(nodemask) * 8,
                             MPOL_MF_STRICT);
    if (mbind_res != 0) {
      return Status::Internal("numa mbind failed");
    }
  }
#endif
  return alloc;
}

Status CpuBackend::Deallocate(const Allocation& alloc) const {
  if (!alloc.ptr) return Status::OK();
  AllocInfo info;
  {
    std::lock_guard<std::mutex> lock(g_alloc_mu);
    g_pool_stats.total_free_calls++;
    auto it = g_allocs.find(alloc.ptr);
    if (it == g_allocs.end()) {
      return Status::Invalid("unknown allocation");
    }
    info = it->second;
    g_allocs.erase(it);
    if (g_pool_stats.in_use_bytes >= info.bytes) {
      g_pool_stats.in_use_bytes -= info.bytes;
    } else {
      g_pool_stats.in_use_bytes = 0;
    }
    if (g_pool_stats.in_use_blocks > 0) {
      g_pool_stats.in_use_blocks--;
    }
  }
  if (*info.pre_guard != kCanary || *info.post_guard != kCanary) {
    return Status::Internal("memory canary corrupted");
  }
#if !defined(_MSC_VER)
  if (info.bytes > 0) {
    std::memset(info.user_ptr, 0, info.bytes);  // scrub before free (best-effort)
    g_pool_stats.scrubbed_bytes += info.bytes;
  }
#endif
  if (info.from_pool && info.bytes <= kPoolMaxSize && info.alignment <= kDefaultAlignment) {
    std::lock_guard<std::mutex> lock(g_alloc_mu);
    g_pool[info.bytes].push_back(info.raw_ptr);
    g_pool_stats.cached_bytes += info.bytes;
    g_pool_stats.cached_blocks++;
  } else {
#if defined(_MSC_VER)
    _aligned_free(info.raw_ptr);
#else
    free(info.raw_ptr);
#endif
  }
  return Status::OK();
}

StatusOr<Allocation> CpuBackend::AllocatePinned(size_t bytes, size_t alignment) const {
  auto alloc_or = Allocate(bytes, alignment);
  if (!alloc_or.ok()) return alloc_or;
  auto alloc = alloc_or.value();
  alloc.kind = AllocationKind::kPinnedHost;
  return alloc;
}

Status CpuBackend::DeallocatePinned(const Allocation& alloc) const {
  return Deallocate(alloc);
}

int CpuBackend::NumThreads() const {
  return static_cast<int>(std::thread::hardware_concurrency());
}

size_t CpuBackend::OutstandingAllocs() const {
  std::lock_guard<std::mutex> lock(g_alloc_mu);
  return g_allocs.size();
}

BackendMemoryStats CpuBackend::MemoryStats() const {
  BackendMemoryStats stats;
  std::lock_guard<std::mutex> lock(g_alloc_mu);
  stats.device = g_pool_stats;
  return stats;
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
      std::string message = "Unknown backend '" + name + "', falling back to '" +
                            BackendNameForEnv(fallback->Type()) + "'";
      std::cerr << message << "\n";
      LogBackend({LogLevel::kWarn, BackendType::kCPU, BackendErrorKind::kInvalidArgument, message,
                  "backend_select"});
      return fallback;
    }
    const Backend* backend = GetBackendByType(requested);
    if (BackendAvailable(backend)) {
      return backend;
    }
    const Backend* fallback = SelectBestAvailableBackend();
    std::string message = "Requested backend '" + name + "' is unavailable; falling back to '" +
                          BackendNameForEnv(fallback->Type()) + "'";
    std::cerr << message << "\n";
    LogBackend({LogLevel::kWarn, BackendType::kCPU, BackendErrorKind::kDiscovery, message,
                "backend_select"});
    return fallback;
  }
  return GetCpuBackend();
}

}  // namespace lattice::runtime
