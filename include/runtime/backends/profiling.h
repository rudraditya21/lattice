#ifndef LATTICE_RUNTIME_BACKENDS_PROFILING_H_
#define LATTICE_RUNTIME_BACKENDS_PROFILING_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>

#include "runtime/backend.h"

namespace lattice::runtime {

class ProfilingState {
   public:
    explicit ProfilingState(BackendType backend);

    void SetEnabled(bool enabled);
    bool Enabled() const;
    void SetHook(ProfilingHook hook);
    ProfilingCounters Snapshot() const;
    void Reset();
    bool ShouldRecord() const;
    void Record(const ProfilingEvent& event);

   private:
    void RecordCounters(const ProfilingEvent& event);
    ProfilingHook CopyHook() const;

    BackendType backend_;
    std::atomic<bool> enabled_{false};
    std::atomic<bool> hook_enabled_{false};
    std::atomic<uint64_t> kernel_launches_{0};
    std::atomic<uint64_t> kernel_launch_ns_{0};
    std::atomic<uint64_t> memcpy_h2d_calls_{0};
    std::atomic<uint64_t> memcpy_h2d_bytes_{0};
    std::atomic<uint64_t> memcpy_h2d_ns_{0};
    std::atomic<uint64_t> memcpy_d2h_calls_{0};
    std::atomic<uint64_t> memcpy_d2h_bytes_{0};
    std::atomic<uint64_t> memcpy_d2h_ns_{0};
    std::atomic<uint64_t> stream_syncs_{0};
    std::atomic<uint64_t> stream_sync_ns_{0};
    std::atomic<uint64_t> event_records_{0};
    std::atomic<uint64_t> event_record_ns_{0};
    std::atomic<uint64_t> event_waits_{0};
    std::atomic<uint64_t> event_wait_ns_{0};
    mutable std::mutex hook_mu_;
    ProfilingHook hook_;
};

class ProfilingScope {
   public:
    ProfilingScope() = default;
    ProfilingScope(ProfilingState* state, ProfilingEvent event);
    ~ProfilingScope();

    void SetStatus(StatusCode status);
    void SetBytes(size_t bytes);
    void SetLabel(std::string label);
    bool active() const { return active_; }

   private:
    ProfilingState* state_ = nullptr;
    ProfilingEvent event_{};
    std::chrono::steady_clock::time_point start_{};
    bool active_ = false;
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_PROFILING_H_
