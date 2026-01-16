#include "test_util.h"

#include <array>
#include <cstdint>

#include "runtime/backend.h"

namespace test {

void RunBackendEdgeTests(TestContext* ctx) {
  auto* backend = const_cast<rt::Backend*>(rt::GetCpuBackend());
  // Alignment and pool reuse behavior.
  auto aligned_or = backend->Allocate(64, 128);
  ExpectTrue(aligned_or.ok(), "cpu_alloc_aligned_status", ctx);
  auto aligned = aligned_or.value();
  ExpectTrue(aligned.ptr != nullptr && aligned.bytes == 64, "cpu_alloc_aligned_ptr", ctx);
  ExpectTrue(aligned.alignment == 128, "cpu_alloc_alignment_field", ctx);
  backend->Deallocate(aligned);

  auto first_or = backend->Allocate(32, 64);
  ExpectTrue(first_or.ok(), "cpu_alloc_pool_first_status", ctx);
  auto first = first_or.value();
  ExpectTrue(!first.from_pool, "cpu_alloc_pool_first_flag", ctx);
  backend->Deallocate(first);

  auto second_or = backend->Allocate(32, 64);
  ExpectTrue(second_or.ok(), "cpu_alloc_pool_second_status", ctx);
  auto second = second_or.value();
  ExpectTrue(second.ptr != nullptr && second.bytes == 32, "cpu_alloc_pool_second_ptr", ctx);
  backend->Deallocate(second);

  // Invalid deallocation should report an error.
  int dummy = 0;
  rt::Allocation bogus;
  bogus.ptr = &dummy;
  bogus.bytes = sizeof(dummy);
  auto invalid = backend->Deallocate(bogus);
  ExpectTrue(invalid.code == rt::StatusCode::kInvalidArgument, "cpu_dealloc_unknown", ctx);

  // Canary corruption should be detected.
  auto corrupt_or = backend->Allocate(8, 64);
  ExpectTrue(corrupt_or.ok(), "cpu_alloc_corrupt_status", ctx);
  auto corrupt = corrupt_or.value();
  auto* bytes = static_cast<uint8_t*>(corrupt.ptr);
  bytes[corrupt.bytes] = 0xAA;
  auto corrupt_status = backend->Deallocate(corrupt);
  ExpectTrue(corrupt_status.code == rt::StatusCode::kInternal, "cpu_canary_detect", ctx);

  // Event readiness and dependency wiring.
  auto event_or = backend->CreateEvent();
  ExpectTrue(event_or.ok(), "cpu_event_create", ctx);
  auto event = event_or.value();
  ExpectTrue(!event->Ready(), "cpu_event_not_ready", ctx);
  event->Record();
  ExpectTrue(event->Ready(), "cpu_event_ready", ctx);
  event->Wait();

  auto stream_or = backend->CreateStream();
  ExpectTrue(stream_or.ok(), "cpu_stream_create_edge", ctx);
  auto stream = stream_or.value();
  stream->AddDependency(event);
  bool ran = false;
  stream->Submit([&]() { ran = true; });
  stream->Synchronize();
  ExpectTrue(ran, "cpu_stream_dependency_runs", ctx);

  auto original_config = backend->GetExecutionConfig();
  rt::ExecutionConfig config = original_config;
  config.sync_on_launch = false;
  config.enable_profiling = true;
  auto set_status = backend->SetExecutionConfig(config);
  ExpectTrue(set_status.ok(), "cpu_exec_config_set", ctx);
  auto roundtrip = backend->GetExecutionConfig();
  ExpectTrue(!roundtrip.sync_on_launch, "cpu_exec_config_sync", ctx);
  ExpectTrue(roundtrip.enable_profiling, "cpu_exec_config_profile", ctx);

  auto start_or = backend->CreateEvent();
  auto end_or = backend->CreateEvent();
  ExpectTrue(start_or.ok() && end_or.ok(), "cpu_exec_event_create", ctx);
  if (start_or.ok() && end_or.ok()) {
    start_or.value()->Record();
    end_or.value()->Record();
    auto elapsed_or = backend->ElapsedNs(start_or.value(), end_or.value());
    ExpectTrue(elapsed_or.ok(), "cpu_elapsed_enabled", ctx);
  }

  backend->ResetProfilingStats();
  std::atomic<int> hook_calls{0};
  backend->SetProfilingHook([&](const rt::ProfilingEvent& ev) {
    if (ev.backend == rt::BackendType::kCPU) {
      hook_calls.fetch_add(1);
    }
  });
  auto prof_stream_or = backend->CreateStream();
  ExpectTrue(prof_stream_or.ok(), "cpu_profile_stream", ctx);
  if (prof_stream_or.ok()) {
    auto prof_stream = prof_stream_or.value();
    auto prof_event_or = prof_stream->CreateEvent();
    ExpectTrue(prof_event_or.ok(), "cpu_profile_event", ctx);
    if (prof_event_or.ok()) {
      auto prof_event = prof_event_or.value();
      prof_stream->RecordEvent(prof_event);
      prof_stream->Synchronize();
      prof_event->Wait();
    }
  }
  auto profile_stats = backend->ProfilingStats();
  ExpectTrue(profile_stats.event_records > 0, "cpu_profile_event_records", ctx);
  ExpectTrue(profile_stats.event_waits > 0, "cpu_profile_event_waits", ctx);
  ExpectTrue(profile_stats.stream_syncs > 0, "cpu_profile_stream_syncs", ctx);
  ExpectTrue(hook_calls.load() > 0, "cpu_profile_hook_calls", ctx);
  backend->ResetProfilingStats();
  auto cleared_stats = backend->ProfilingStats();
  ExpectTrue(cleared_stats.event_records == 0, "cpu_profile_reset_records", ctx);
  backend->SetProfilingHook(rt::ProfilingHook{});

  config.enable_profiling = false;
  backend->SetExecutionConfig(config);
  if (start_or.ok() && end_or.ok()) {
    start_or.value()->Record();
    end_or.value()->Record();
    auto elapsed_or = backend->ElapsedNs(start_or.value(), end_or.value());
    ExpectTrue(elapsed_or.status().code == rt::StatusCode::kUnavailable, "cpu_elapsed_disabled",
               ctx);
  }
  backend->SetExecutionConfig(original_config);

  const std::array<rt::BackendType, 4> gpu_types = {
      rt::BackendType::kOpenCL, rt::BackendType::kCUDA, rt::BackendType::kHIP,
      rt::BackendType::kMetal};
  for (auto type : gpu_types) {
    const auto* gpu = rt::GetBackendByType(type);
    if (!gpu) continue;
    auto gpu_stream_or = gpu->CreateStream();
    if (!gpu_stream_or.ok()) continue;
    auto gpu_alloc_or = gpu->Allocate(16, 64);
    ExpectTrue(gpu_alloc_or.ok(), "gpu_alloc_status", ctx);
    if (gpu_alloc_or.ok()) {
      auto status = gpu->Deallocate(gpu_alloc_or.value());
      ExpectTrue(status.ok(), "gpu_dealloc_status", ctx);
    }
    auto gpu_event_or = gpu->CreateEvent();
    ExpectTrue(gpu_event_or.ok(), "gpu_event_status", ctx);
    if (gpu_event_or.ok()) {
      gpu_event_or.value()->Record();
      gpu_event_or.value()->Wait();
    }
  }
}

}  // namespace test
