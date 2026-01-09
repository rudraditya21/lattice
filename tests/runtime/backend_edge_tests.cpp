#include "test_util.h"

#include <array>
#include <cstdint>

#include "runtime/backend.h"

namespace test {

void RunBackendEdgeTests(TestContext* ctx) {
  const auto* backend = rt::GetCpuBackend();
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
