#include "runtime/backends/profiling.h"

#include <chrono>

namespace lattice::runtime {

namespace {

uint64_t DurationNs(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end) {
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count();
    if (ns < 0)
        ns = 0;
    return static_cast<uint64_t>(ns);
}

}  // namespace

ProfilingState::ProfilingState(BackendType backend) : backend_(backend) {}

void ProfilingState::SetEnabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_relaxed);
}

bool ProfilingState::Enabled() const {
    return enabled_.load(std::memory_order_relaxed);
}

void ProfilingState::SetHook(ProfilingHook hook) {
    std::lock_guard<std::mutex> lock(hook_mu_);
    hook_ = std::move(hook);
    hook_enabled_.store(static_cast<bool>(hook_), std::memory_order_release);
}

ProfilingHook ProfilingState::CopyHook() const {
    if (!hook_enabled_.load(std::memory_order_acquire)) {
        return {};
    }
    std::lock_guard<std::mutex> lock(hook_mu_);
    return hook_;
}

bool ProfilingState::ShouldRecord() const {
    return enabled_.load(std::memory_order_relaxed) ||
           hook_enabled_.load(std::memory_order_relaxed);
}

void ProfilingState::Record(const ProfilingEvent& event) {
    if (!ShouldRecord()) {
        return;
    }
    ProfilingEvent out = event;
    out.backend = backend_;
    RecordCounters(out);
    ProfilingHook hook = CopyHook();
    if (hook) {
        hook(out);
    }
}

void ProfilingState::RecordCounters(const ProfilingEvent& event) {
    switch (event.kind) {
        case ProfilingEventKind::kKernelLaunch:
            kernel_launches_.fetch_add(1, std::memory_order_relaxed);
            kernel_launch_ns_.fetch_add(event.duration_ns,
                                        std::memory_order_relaxed);
            break;
        case ProfilingEventKind::kMemcpyH2D:
            memcpy_h2d_calls_.fetch_add(1, std::memory_order_relaxed);
            memcpy_h2d_bytes_.fetch_add(static_cast<uint64_t>(event.bytes),
                                        std::memory_order_relaxed);
            memcpy_h2d_ns_.fetch_add(event.duration_ns,
                                     std::memory_order_relaxed);
            break;
        case ProfilingEventKind::kMemcpyD2H:
            memcpy_d2h_calls_.fetch_add(1, std::memory_order_relaxed);
            memcpy_d2h_bytes_.fetch_add(static_cast<uint64_t>(event.bytes),
                                        std::memory_order_relaxed);
            memcpy_d2h_ns_.fetch_add(event.duration_ns,
                                     std::memory_order_relaxed);
            break;
        case ProfilingEventKind::kStreamSync:
            stream_syncs_.fetch_add(1, std::memory_order_relaxed);
            stream_sync_ns_.fetch_add(event.duration_ns,
                                      std::memory_order_relaxed);
            break;
        case ProfilingEventKind::kEventRecord:
            event_records_.fetch_add(1, std::memory_order_relaxed);
            event_record_ns_.fetch_add(event.duration_ns,
                                       std::memory_order_relaxed);
            break;
        case ProfilingEventKind::kEventWait:
            event_waits_.fetch_add(1, std::memory_order_relaxed);
            event_wait_ns_.fetch_add(event.duration_ns,
                                     std::memory_order_relaxed);
            break;
    }
}

ProfilingCounters ProfilingState::Snapshot() const {
    ProfilingCounters out;
    out.kernel_launches = kernel_launches_.load(std::memory_order_relaxed);
    out.kernel_launch_ns = kernel_launch_ns_.load(std::memory_order_relaxed);
    out.memcpy_h2d_calls = memcpy_h2d_calls_.load(std::memory_order_relaxed);
    out.memcpy_h2d_bytes = memcpy_h2d_bytes_.load(std::memory_order_relaxed);
    out.memcpy_h2d_ns = memcpy_h2d_ns_.load(std::memory_order_relaxed);
    out.memcpy_d2h_calls = memcpy_d2h_calls_.load(std::memory_order_relaxed);
    out.memcpy_d2h_bytes = memcpy_d2h_bytes_.load(std::memory_order_relaxed);
    out.memcpy_d2h_ns = memcpy_d2h_ns_.load(std::memory_order_relaxed);
    out.stream_syncs = stream_syncs_.load(std::memory_order_relaxed);
    out.stream_sync_ns = stream_sync_ns_.load(std::memory_order_relaxed);
    out.event_records = event_records_.load(std::memory_order_relaxed);
    out.event_record_ns = event_record_ns_.load(std::memory_order_relaxed);
    out.event_waits = event_waits_.load(std::memory_order_relaxed);
    out.event_wait_ns = event_wait_ns_.load(std::memory_order_relaxed);
    return out;
}

void ProfilingState::Reset() {
    kernel_launches_.store(0, std::memory_order_relaxed);
    kernel_launch_ns_.store(0, std::memory_order_relaxed);
    memcpy_h2d_calls_.store(0, std::memory_order_relaxed);
    memcpy_h2d_bytes_.store(0, std::memory_order_relaxed);
    memcpy_h2d_ns_.store(0, std::memory_order_relaxed);
    memcpy_d2h_calls_.store(0, std::memory_order_relaxed);
    memcpy_d2h_bytes_.store(0, std::memory_order_relaxed);
    memcpy_d2h_ns_.store(0, std::memory_order_relaxed);
    stream_syncs_.store(0, std::memory_order_relaxed);
    stream_sync_ns_.store(0, std::memory_order_relaxed);
    event_records_.store(0, std::memory_order_relaxed);
    event_record_ns_.store(0, std::memory_order_relaxed);
    event_waits_.store(0, std::memory_order_relaxed);
    event_wait_ns_.store(0, std::memory_order_relaxed);
}

ProfilingScope::ProfilingScope(ProfilingState* state, ProfilingEvent event)
    : state_(state), event_(std::move(event)) {
    if (!state_ || !state_->ShouldRecord()) {
        state_ = nullptr;
        return;
    }
    active_ = true;
    start_ = std::chrono::steady_clock::now();
}

ProfilingScope::~ProfilingScope() {
    if (!active_ || !state_) {
        return;
    }
    auto end = std::chrono::steady_clock::now();
    event_.duration_ns = DurationNs(start_, end);
    state_->Record(event_);
}

void ProfilingScope::SetStatus(StatusCode status) {
    if (!active_) {
        return;
    }
    event_.status = status;
}

void ProfilingScope::SetBytes(size_t bytes) {
    if (!active_) {
        return;
    }
    event_.bytes = bytes;
}

void ProfilingScope::SetLabel(std::string label) {
    if (!active_) {
        return;
    }
    event_.label = std::move(label);
}

}  // namespace lattice::runtime
