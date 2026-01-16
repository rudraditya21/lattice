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
    return base;
}

}  // namespace lattice::runtime
