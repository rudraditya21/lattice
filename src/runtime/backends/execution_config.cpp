#include "runtime/backends/execution_config.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace lattice::runtime {
namespace {

bool ParseBoolEnv(const char* value, bool* out) {
    if (!value || !out)
        return false;
    std::string normalized(value);
    std::transform(
        normalized.begin(), normalized.end(), normalized.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (normalized == "1" || normalized == "true" || normalized == "yes" ||
        normalized == "on") {
        *out = true;
        return true;
    }
    if (normalized == "0" || normalized == "false" || normalized == "no" ||
        normalized == "off") {
        *out = false;
        return true;
    }
    return false;
}

}  // namespace

ExecutionConfig LoadExecutionConfig(const std::string& prefix,
                                    ExecutionConfig base) {
    bool value = false;
    const std::string sync_key = prefix + "_SYNC_ON_LAUNCH";
    if (ParseBoolEnv(std::getenv(sync_key.c_str()), &value)) {
        base.sync_on_launch = value;
    }
    const std::string async_key = prefix + "_ASYNC_LAUNCH";
    if (ParseBoolEnv(std::getenv(async_key.c_str()), &value)) {
        base.sync_on_launch = !value;
    }
    const std::string fast_math_key = prefix + "_FAST_MATH";
    if (ParseBoolEnv(std::getenv(fast_math_key.c_str()), &value)) {
        base.enable_fast_math = value;
    }
    const std::string vectorize_key = prefix + "_VECTORIZE";
    if (ParseBoolEnv(std::getenv(vectorize_key.c_str()), &value)) {
        base.enable_vectorize = value;
    }
    const std::string mixed_key = prefix + "_MIXED_PRECISION";
    if (ParseBoolEnv(std::getenv(mixed_key.c_str()), &value)) {
        base.enable_mixed_precision = value;
    }
    const std::string trace_key = prefix + "_TRACE_KERNELS";
    if (ParseBoolEnv(std::getenv(trace_key.c_str()), &value)) {
        base.trace_kernels = value;
    }
    const std::string timeout_key = prefix + "_KERNEL_TIMEOUT_MS";
    if (const char* env = std::getenv(timeout_key.c_str())) {
        char* end = nullptr;
        long long parsed = std::strtoll(env, &end, 10);
        if (end != env) {
            if (parsed < 0)
                parsed = 0;
            base.kernel_timeout_ms = static_cast<int64_t>(parsed);
        }
    }
    return base;
}

}  // namespace lattice::runtime
