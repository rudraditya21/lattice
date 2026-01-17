#include "test_util.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <string>

#include "runtime/backends/cuda_backend.h"
#include "runtime/backends/hip_backend.h"
#include "runtime/backends/opencl_backend.h"
#if defined(__APPLE__)
#include "runtime/backends/metal_backend.h"
#endif

namespace test {

namespace {

bool EnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return false;
  }
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

void RunSmokeLoop(const std::string& name,
                  const rt::Backend* backend,
                  const std::function<rt::Status()>& smoke,
                  TestContext* ctx) {
  if (!rt::BackendAvailable(backend)) {
    return;
  }
  const int iterations = 3;
  for (int i = 0; i < iterations; ++i) {
    rt::Status status = smoke();
    ExpectTrue(status.ok(),
               name + "_smoke_iter_" + std::to_string(i), ctx);
  }
  ExpectTrue(backend->OutstandingAllocs() == 0,
             name + "_no_leaks", ctx);

  for (int i = 0; i < 32; ++i) {
    auto alloc_or = backend->Allocate(2048, 64);
    ExpectTrue(alloc_or.ok(),
               name + "_alloc_iter_" + std::to_string(i), ctx);
    if (alloc_or.ok()) {
      auto st = backend->Deallocate(alloc_or.value());
      ExpectTrue(st.ok(),
                 name + "_dealloc_iter_" + std::to_string(i), ctx);
    }
  }
  ExpectTrue(backend->OutstandingAllocs() == 0,
             name + "_no_leaks_after_alloc", ctx);
}

}  // namespace

void RunGpuStressTests(TestContext* ctx) {
  if (!EnvEnabled("LATTICE_GPU_STRESS_TEST")) {
    return;
  }

  RunSmokeLoop("opencl", rt::GetBackendByType(rt::BackendType::kOpenCL),
               [] { return rt::RunOpenCLSmokeTest(); }, ctx);
  RunSmokeLoop("cuda", rt::GetBackendByType(rt::BackendType::kCUDA),
               [] { return rt::RunCudaSmokeTest(); }, ctx);
  RunSmokeLoop("hip", rt::GetBackendByType(rt::BackendType::kHIP),
               [] { return rt::RunHipSmokeTest(); }, ctx);
#if defined(__APPLE__)
  RunSmokeLoop("metal", rt::GetBackendByType(rt::BackendType::kMetal),
               [] { return rt::RunMetalSmokeTest(); }, ctx);
#endif
}

}  // namespace test
