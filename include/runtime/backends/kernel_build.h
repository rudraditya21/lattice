#ifndef LATTICE_RUNTIME_BACKENDS_KERNEL_BUILD_H_
#define LATTICE_RUNTIME_BACKENDS_KERNEL_BUILD_H_

#include <cstdint>
#include <string>

#include "runtime/backend.h"

namespace lattice::runtime {

struct KernelBuildDefines {
    BackendType backend = BackendType::kCPU;
    int device_index = -1;
    uint32_t abi_version = 0;
    uint32_t abi_version_min = 0;
    bool has_fp16 = false;
    bool has_fp64 = false;
    uint64_t device_type = 0;
    uint64_t vendor_id = 0;
};

std::string KernelDefineString(const KernelBuildDefines& defs,
                               const std::string& prefix = "-D");
std::string LoadBuildOptionsEnv(const std::string& backend_env);

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_KERNEL_BUILD_H_
