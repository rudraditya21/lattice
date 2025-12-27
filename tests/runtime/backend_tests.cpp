#include "runtime/backend.h"
#include "test_util.h"

namespace test {

void RunBackendTests(TestContext* ctx) {
  const auto* backend = rt::GetDefaultBackend();
  ExpectTrue(backend->Type() == rt::BackendType::kCPU, "backend_type_cpu", ctx);
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
  auto dealloc_status = backend->Deallocate(alloc);
  ExpectTrue(dealloc_status.ok(), "backend_dealloc_status", ctx);
  ExpectTrue(backend->OutstandingAllocs() == 0, "backend_no_leaks", ctx);
}

}  // namespace test
