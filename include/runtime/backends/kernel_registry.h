#ifndef LATTICE_RUNTIME_BACKENDS_KERNEL_REGISTRY_H_
#define LATTICE_RUNTIME_BACKENDS_KERNEL_REGISTRY_H_

#include <string_view>
#include <vector>

namespace lattice::runtime {

struct KernelDefinition {
    std::string_view name;
    std::string_view file;
};

const KernelDefinition* FindKernelDefinition(std::string_view name);
std::vector<KernelDefinition> AllKernelDefinitions();

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_KERNEL_REGISTRY_H_
