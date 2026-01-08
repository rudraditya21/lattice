#include "runtime/backends/device_quirks.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <vector>

namespace lattice::runtime {

namespace {

enum class BackendScope { kAny, kOpenCL, kCUDA, kHIP, kMetal };

struct DeviceQuirkEntry {
  BackendScope scope;
  const char* vendor_substr;
  const char* name_substr;
  const char* driver_substr;
  uint32_t flags;
  bool disabled;
  const char* reason;
};

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

bool Contains(const std::string& haystack, const char* needle) {
  if (!needle || !needle[0]) return true;
  return haystack.find(needle) != std::string::npos;
}

bool MatchesScope(BackendScope scope, BackendType backend) {
  switch (scope) {
    case BackendScope::kAny:
      return true;
    case BackendScope::kOpenCL:
      return backend == BackendType::kOpenCL;
    case BackendScope::kCUDA:
      return backend == BackendType::kCUDA;
    case BackendScope::kHIP:
      return backend == BackendType::kHIP;
    case BackendScope::kMetal:
      return backend == BackendType::kMetal;
  }
  return false;
}

std::vector<std::string> SplitPatterns(const char* value) {
  std::vector<std::string> out;
  if (!value) return out;
  std::string current;
  for (const char c : std::string(value)) {
    if (c == ',' || c == ';') {
      if (!current.empty()) {
        out.push_back(ToLower(current));
        current.clear();
      }
      continue;
    }
    if (!std::isspace(static_cast<unsigned char>(c))) {
      current.push_back(c);
    }
  }
  if (!current.empty()) out.push_back(ToLower(current));
  return out;
}

const DeviceQuirkEntry kQuirkTable[] = {
    {BackendScope::kOpenCL, "portable computing language", "", "", kSoftwareEmulation, false,
     "OpenCL POCL (software)"},
    {BackendScope::kOpenCL, "mesa", "llvmpipe", "", kSoftwareEmulation, false,
     "OpenCL Mesa llvmpipe (software)"},
    {BackendScope::kOpenCL, "mesa", "lavapipe", "", kSoftwareEmulation, false,
     "OpenCL Mesa lavapipe (software)"},
};

}  // namespace

DeviceQuirkInfo QueryDeviceQuirks(BackendType backend, const std::string& vendor,
                                  const std::string& name, const std::string& driver) {
  DeviceQuirkInfo info;
  const std::string vendor_lc = ToLower(vendor);
  const std::string name_lc = ToLower(name);
  const std::string driver_lc = ToLower(driver);

  const char* ignore = std::getenv("LATTICE_IGNORE_DEVICE_QUIRKS");
  const bool ignore_quirks = ignore && ignore[0] != '\0';

  if (!ignore_quirks) {
    for (const auto& entry : kQuirkTable) {
      if (!MatchesScope(entry.scope, backend)) continue;
      if (!Contains(vendor_lc, entry.vendor_substr)) continue;
      if (!Contains(name_lc, entry.name_substr)) continue;
      if (!Contains(driver_lc, entry.driver_substr)) continue;
      info.flags |= entry.flags;
      if (entry.disabled) {
        info.disabled = true;
        if (info.reason.empty()) info.reason = entry.reason ? entry.reason : "";
      }
    }
  }

  const char* disable_software = std::getenv("LATTICE_DISABLE_SOFTWARE_DEVICES");
  if (disable_software && disable_software[0] != '\0') {
    if (info.flags & kSoftwareEmulation) {
      info.disabled = true;
      if (info.reason.empty()) info.reason = "software device disabled";
    }
  }

  const auto blacklist = SplitPatterns(std::getenv("LATTICE_DEVICE_BLACKLIST"));
  for (const auto& pattern : blacklist) {
    if (pattern.empty()) continue;
    if (vendor_lc.find(pattern) != std::string::npos ||
        name_lc.find(pattern) != std::string::npos ||
        driver_lc.find(pattern) != std::string::npos) {
      info.disabled = true;
      info.reason = "blacklisted by LATTICE_DEVICE_BLACKLIST";
      break;
    }
  }

  return info;
}

}  // namespace lattice::runtime
