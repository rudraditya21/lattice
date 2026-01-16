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

  rt::KernelDispatchKey key;
  key.backend = rt::BackendType::kCUDA;
  key.vendor = rt::KernelVendor::kNvidia;
  key.arch_major = 8;
  key.vectorize = true;
  key.use_fp64 = false;
  const auto* matmul = rt::SelectKernelDefinition(rt::KernelOp::kMatmul, key);
  ExpectTrue(matmul != nullptr, "kernel_registry_select_matmul", ctx);
  if (matmul) {
    ExpectTrue(matmul->name == "lattice_matmul_t32",
               "kernel_registry_matmul_variant", ctx);
  }

  const auto* elemwise =
      rt::SelectKernelDefinition(rt::KernelOp::kElemwiseAdd, key);
  ExpectTrue(elemwise != nullptr, "kernel_registry_select_elemwise", ctx);
  if (elemwise) {
    ExpectTrue(elemwise->name == "lattice_elemwise_add_vec4",
               "kernel_registry_elemwise_variant", ctx);
  }
}

}  // namespace test
