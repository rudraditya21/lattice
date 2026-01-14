#include "runtime/backends/kernel_build.h"

#include <cstdlib>
#include <sstream>

namespace lattice::runtime {

namespace {

const char* BackendMacro(BackendType backend) {
    switch (backend) {
        case BackendType::kOpenCL:
            return "LATTICE_BACKEND_OPENCL";
        case BackendType::kCUDA:
            return "LATTICE_BACKEND_CUDA";
        case BackendType::kHIP:
            return "LATTICE_BACKEND_HIP";
        case BackendType::kMetal:
            return "LATTICE_BACKEND_METAL";
        case BackendType::kCPU:
            return "LATTICE_BACKEND_CPU";
    }
    return "LATTICE_BACKEND_CPU";
}

void AppendDefine(std::ostringstream& out,
                  const std::string& prefix,
                  const std::string& name,
                  const std::string& value) {
    if (!prefix.empty() && !name.empty()) {
        out << prefix << name;
        if (!value.empty()) {
            out << "=" << value;
        }
    }
}

std::string GetEnvString(const char* key) {
    if (!key)
        return "";
    const char* env = std::getenv(key);
    if (!env || env[0] == '\0')
        return "";
    return std::string(env);
}

}  // namespace

std::string KernelDefineString(const KernelBuildDefines& defs,
                               const std::string& prefix) {
    std::ostringstream out;
    bool first = true;
    auto add = [&](const std::string& name, const std::string& value = "") {
        if (!first)
            out << " ";
        first = false;
        AppendDefine(out, prefix, name, value);
    };

    add(BackendMacro(defs.backend), "1");
    if (defs.device_index >= 0) {
        add("LATTICE_DEVICE_INDEX", std::to_string(defs.device_index));
    }
    if (defs.device_type != 0) {
        add("LATTICE_DEVICE_TYPE", std::to_string(defs.device_type));
    }
    if (defs.vendor_id != 0) {
        add("LATTICE_VENDOR_ID", std::to_string(defs.vendor_id));
    }
    if (defs.abi_version != 0) {
        add("LATTICE_ABI_VERSION", std::to_string(defs.abi_version));
    }
    if (defs.abi_version_min != 0) {
        add("LATTICE_ABI_VERSION_MIN", std::to_string(defs.abi_version_min));
    }
    if (defs.has_fp16) {
        add("LATTICE_HAS_FP16", "1");
    }
    if (defs.has_fp64) {
        add("LATTICE_HAS_FP64", "1");
    }
    return out.str();
}

std::string LoadBuildOptionsEnv(const std::string& backend_env) {
    std::string global = GetEnvString("LATTICE_BUILD_OPTIONS");
    std::string backend = GetEnvString(backend_env.c_str());
    if (global.empty())
        return backend;
    if (backend.empty())
        return global;
    return global + " " + backend;
}

}  // namespace lattice::runtime
