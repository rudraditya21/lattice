#include "runtime/backends/kernel_registry.h"
#include "test_util.h"

namespace test {

void RunKernelRegistryTests(TestContext* ctx) {
  const auto* known = rt::FindKernelDefinition("lattice_elemwise_add");
  ExpectTrue(known != nullptr, "kernel_registry_known", ctx);
  if (known) {
    ExpectTrue(known->file == "tensor_elemwise_add", "kernel_registry_file", ctx);
  }

  const auto* missing = rt::FindKernelDefinition("lattice_missing");
  ExpectTrue(missing == nullptr, "kernel_registry_missing", ctx);

  auto all = rt::AllKernelDefinitions();
  ExpectTrue(!all.empty(), "kernel_registry_all", ctx);
}

}  // namespace test
