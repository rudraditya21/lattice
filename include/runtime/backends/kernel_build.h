#ifndef LATTICE_RUNTIME_BACKENDS_KERNEL_BUILD_H_
#define LATTICE_RUNTIME_BACKENDS_KERNEL_BUILD_H_

#include <cstddef>
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
    bool fast_math = false;
    bool vectorize = false;
    bool mixed_precision = false;
    uint64_t device_type = 0;
    uint64_t vendor_id = 0;
    uint32_t arch_major = 0;
    uint32_t arch_minor = 0;
    uint32_t vector_width = 0;
};

std::string KernelDefineString(const KernelBuildDefines& defs,
                               const std::string& prefix = "-D");
std::string LoadBuildOptionsEnv(const std::string& backend_env);
bool KernelDebugEnabled(const std::string& backend_env);
bool KernelRebuildOnFailureEnabled(const std::string& backend_env);
std::string KernelDebugOptions(BackendType backend);
std::string Sha256Hex(const void* data, size_t size);
std::string Sha256Hex(const std::string& data);
std::string NormalizePathArg(const std::string& path);
std::string BuildIncludeOption(const std::string& include_dir,
                               const std::string& prefix = "-I");
void AppendOption(std::string* out, const std::string& option);

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_KERNEL_BUILD_H_
