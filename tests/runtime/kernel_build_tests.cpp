#include "test_util.h"

#include "runtime/backends/kernel_build.h"

namespace test {

void RunKernelBuildTests(TestContext* ctx) {
  ExpectTrue(rt::KernelRebuildOnFailureEnabled("LATTICE_CUDA_REBUILD_ON_FAILURE"),
             "kernel_rebuild_default", ctx);

  {
    ScopedEnvVar global("LATTICE_REBUILD_ON_FAILURE", "0");
    ExpectTrue(!rt::KernelRebuildOnFailureEnabled("LATTICE_CUDA_REBUILD_ON_FAILURE"),
               "kernel_rebuild_global_disable", ctx);
  }

  {
    ScopedEnvVar global("LATTICE_REBUILD_ON_FAILURE", "0");
    ScopedEnvVar backend("LATTICE_CUDA_REBUILD_ON_FAILURE", "1");
    ExpectTrue(rt::KernelRebuildOnFailureEnabled("LATTICE_CUDA_REBUILD_ON_FAILURE"),
               "kernel_rebuild_backend_override", ctx);
  }

  {
    ScopedEnvVar global("LATTICE_REBUILD_ON_FAILURE", "1");
    ScopedEnvVar backend("LATTICE_CUDA_REBUILD_ON_FAILURE", "0");
    ExpectTrue(!rt::KernelRebuildOnFailureEnabled("LATTICE_CUDA_REBUILD_ON_FAILURE"),
               "kernel_rebuild_backend_disable", ctx);
  }
}

}  // namespace test
