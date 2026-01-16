#ifndef LATTICE_RUNTIME_BACKENDS_KERNEL_REGISTRY_H_
#define LATTICE_RUNTIME_BACKENDS_KERNEL_REGISTRY_H_

#include <string_view>
#include <vector>

#include "runtime/backend.h"

namespace lattice::runtime {

struct KernelDefinition {
    std::string_view name;
    std::string_view file;
};

enum class KernelOp {
    kElemwiseAdd,
    kElemwiseSub,
    kElemwiseMul,
    kElemwiseDiv,
    kReduceSum,
    kReduceMean,
    kReduceVar,
    kReduceStd,
    kTranspose,
    kMatmul,
    kConv2d,
};

enum class KernelVendor { kAny, kNvidia, kAmd, kIntel, kApple };

struct KernelDispatchKey {
    BackendType backend = BackendType::kCPU;
    KernelVendor vendor = KernelVendor::kAny;
    uint32_t arch_major = 0;
    uint32_t arch_minor = 0;
    bool use_fp64 = false;
    bool vectorize = false;
};

const KernelDefinition* FindKernelDefinition(std::string_view name);
const KernelDefinition* SelectKernelDefinition(KernelOp op,
                                               const KernelDispatchKey& key);
std::vector<KernelDefinition> AllKernelDefinitions();

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_KERNEL_REGISTRY_H_
