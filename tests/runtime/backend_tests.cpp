#include "runtime/backend.h"
#include "test_util.h"

#include <cstdlib>

namespace test {

void RunBackendTests(TestContext* ctx) {
  const auto* backend = rt::GetDefaultBackend();
  ExpectTrue(backend->Type() == rt::BackendType::kCPU ||
                 backend->Type() == rt::BackendType::kOpenCL ||
                 backend->Type() == rt::BackendType::kCUDA ||
                 backend->Type() == rt::BackendType::kHIP ||
                 backend->Type() == rt::BackendType::kMetal,
             "backend_type_supported", ctx);
  auto caps = backend->Capabilities();
  ExpectTrue(caps.supports_dense, "backend_supports_dense", ctx);

  auto stream_or = backend->CreateStream();
  ExpectTrue(stream_or.ok(), "backend_stream_status", ctx);
  auto stream = stream_or.value();
  bool ran = false;
  stream->Submit([&]() { ran = true; });
  stream->Synchronize();
  ExpectTrue(ran, "backend_stream_runs_task", ctx);

  // Allocation/deallocation
  auto alloc_or = backend->Allocate(128, 64);
  ExpectTrue(alloc_or.ok(), "backend_alloc_status", ctx);
  auto alloc = alloc_or.value();
  if (backend->Type() == rt::BackendType::kCPU) {
    ExpectTrue(alloc.ptr != nullptr && alloc.bytes == 128, "backend_alloc", ctx);
    // Newly allocated memory should be zeroed.
    bool all_zero = true;
    auto* bytes = static_cast<uint8_t*>(alloc.ptr);
    for (size_t i = 0; i < alloc.bytes; ++i) {
      if (bytes[i] != 0) {
        all_zero = false;
        break;
      }
    }
    ExpectTrue(all_zero, "backend_alloc_zeroed", ctx);
  } else {
    ExpectTrue(alloc.device_handle != nullptr && alloc.bytes == 128, "backend_alloc_device", ctx);
  }
  auto dealloc_status = backend->Deallocate(alloc);
  ExpectTrue(dealloc_status.ok(), "backend_dealloc_status", ctx);
  ExpectTrue(backend->OutstandingAllocs() == 0, "backend_no_leaks", ctx);

  const char* run_smoke = std::getenv("LATTICE_GPU_SMOKE_TEST");
  if (run_smoke && run_smoke[0] != '\0') {
    rt::Status status = rt::Status::OK();
    switch (backend->Type()) {
      case rt::BackendType::kOpenCL:
        status = rt::RunOpenCLSmokeTest();
        break;
      case rt::BackendType::kCUDA:
        status = rt::RunCudaSmokeTest();
        break;
      case rt::BackendType::kHIP:
        status = rt::RunHipSmokeTest();
        break;
      case rt::BackendType::kMetal:
#if defined(__APPLE__)
        status = rt::RunMetalSmokeTest();
#else
        status = rt::Status::Unavailable("Metal backend not supported");
#endif
        break;
      case rt::BackendType::kCPU:
        status = rt::Status::OK();
        break;
    }
    ExpectTrue(status.ok(), "gpu_smoke_test", ctx);
  }
}

}  // namespace test
