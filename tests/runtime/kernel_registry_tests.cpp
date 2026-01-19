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
  const auto* conv2d = rt::SelectKernelDefinition(rt::KernelOp::kConv2d, key);
  ExpectTrue(conv2d != nullptr, "kernel_registry_select_conv2d", ctx);
  if (conv2d) {
    ExpectTrue(conv2d->name == "lattice_conv2d_t16",
               "kernel_registry_conv2d_variant", ctx);
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
