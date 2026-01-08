#include "runtime/backends/device_selector.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <unordered_set>

namespace lattice::runtime {

namespace {

std::string ToLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
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

const char* GetEnvWithFallback(const char* prefix, const char* suffix) {
  if (!prefix || !suffix) return nullptr;
  const std::string name = std::string(prefix) + suffix;
  const char* value = std::getenv(name.c_str());
  if (value && value[0] != '\0') return value;
  const std::string global = std::string("LATTICE") + suffix;
  value = std::getenv(global.c_str());
  return value;
}

DeviceKind ParseKind(const char* value) {
  if (!value) return DeviceKind::kAny;
  std::string v = ToLower(value);
  if (v == "cpu") return DeviceKind::kCPU;
  if (v == "gpu") return DeviceKind::kGPU;
  if (v == "accel" || v == "accelerator") return DeviceKind::kAccelerator;
  return DeviceKind::kAny;
}

bool MatchesPatterns(const DeviceIdentity& dev, const std::vector<std::string>& patterns) {
  if (patterns.empty()) return true;
  const std::string name = ToLower(dev.name);
  const std::string vendor = ToLower(dev.vendor);
  const std::string driver = ToLower(dev.driver);
  for (const auto& pattern : patterns) {
    if (pattern.empty()) continue;
    if (name.find(pattern) != std::string::npos || vendor.find(pattern) != std::string::npos ||
        driver.find(pattern) != std::string::npos) {
      return true;
    }
  }
  return false;
}

std::vector<int> ParseMask(const std::string& mask, size_t count) {
  std::vector<int> out;
  if (mask.empty()) return out;
  size_t idx = 0;
  for (char c : mask) {
    if (c == '0' || c == '1') {
      if (idx >= count) break;
      if (c == '1') out.push_back(static_cast<int>(idx));
      ++idx;
    }
  }
  return out;
}

std::vector<int> UniquePreserveOrder(const std::vector<int>& values) {
  std::vector<int> out;
  std::unordered_set<int> seen;
  for (int v : values) {
    if (seen.insert(v).second) out.push_back(v);
  }
  return out;
}

std::vector<int> Reorder(const std::vector<int>& selected, const std::vector<int>& order) {
  if (order.empty()) return selected;
  std::unordered_set<int> selected_set(selected.begin(), selected.end());
  std::vector<int> out;
  out.reserve(selected.size());
  for (int idx : order) {
    if (selected_set.erase(idx) > 0) {
      out.push_back(idx);
    }
  }
  for (int idx : selected) {
    if (selected_set.erase(idx) > 0) {
      out.push_back(idx);
    }
  }
  return out;
}

std::string BuildDiagnostics(const DeviceSelectionOptions& options) {
  std::ostringstream ss;
  ss << "No devices matched selection (";
  bool first = true;
  auto add = [&](const std::string& label, const std::string& value) {
    if (value.empty()) return;
    if (!first) ss << ", ";
    first = false;
    ss << label << "=" << value;
  };
  auto join = [](const std::vector<int>& values) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
      if (i) out << ",";
      out << values[i];
    }
    return out.str();
  };
  add("type", options.kind == DeviceKind::kCPU           ? "cpu"
              : options.kind == DeviceKind::kGPU         ? "gpu"
              : options.kind == DeviceKind::kAccelerator ? "accel"
                                                         : "any");
  if (!options.include_patterns.empty()) {
    add("include", "patterns");
  }
  if (!options.exclude_patterns.empty()) {
    add("exclude", "patterns");
  }
  add("indices", join(options.indices));
  add("order", join(options.order));
  add("mask", options.mask);
  ss << ")";
  return ss.str();
}

}  // namespace

DeviceSelectionOptions LoadDeviceSelectionOptions(const char* prefix) {
  DeviceSelectionOptions options;
  const char* type_env = GetEnvWithFallback(prefix, "_DEVICE_TYPE");
  options.kind = ParseKind(type_env);
  options.include_patterns = SplitPatterns(GetEnvWithFallback(prefix, "_DEVICE_INCLUDE"));
  options.exclude_patterns = SplitPatterns(GetEnvWithFallback(prefix, "_DEVICE_EXCLUDE"));
  if (const char* mask = GetEnvWithFallback(prefix, "_DEVICE_MASK")) {
    options.mask = mask;
  }
  if (const char* indices = GetEnvWithFallback(prefix, "_DEVICE_INDICES")) {
    options.indices = ParseIndexList(indices);
  }
  if (const char* order = GetEnvWithFallback(prefix, "_DEVICE_ORDER")) {
    options.order = ParseIndexList(order);
  }
  options.indices = UniquePreserveOrder(options.indices);
  options.order = UniquePreserveOrder(options.order);
  options.explicit_selection = options.kind != DeviceKind::kAny ||
                               !options.include_patterns.empty() ||
                               !options.exclude_patterns.empty() || !options.mask.empty() ||
                               !options.indices.empty() || !options.order.empty();
  return options;
}

DeviceSelectionResult SelectDevices(const std::vector<DeviceIdentity>& devices,
                                    const DeviceSelectionOptions& options) {
  DeviceSelectionResult result;
  const auto mask_indices = ParseMask(options.mask, devices.size());
  std::unordered_set<int> mask_set(mask_indices.begin(), mask_indices.end());
  std::vector<int> selected;
  selected.reserve(devices.size());
  for (const auto& dev : devices) {
    if (options.kind != DeviceKind::kAny && dev.kind != options.kind) continue;
    if (!options.include_patterns.empty() && !MatchesPatterns(dev, options.include_patterns)) {
      continue;
    }
    if (!options.exclude_patterns.empty() && MatchesPatterns(dev, options.exclude_patterns)) {
      continue;
    }
    if (!options.indices.empty() && std::find(options.indices.begin(), options.indices.end(),
                                              dev.index) == options.indices.end()) {
      continue;
    }
    if (!options.mask.empty() && mask_set.find(dev.index) == mask_set.end()) {
      continue;
    }
    selected.push_back(dev.index);
  }

  if (!options.order.empty()) {
    selected = Reorder(selected, options.order);
  } else if (!options.indices.empty()) {
    selected = Reorder(selected, options.indices);
  }

  result.indices = selected;
  if (selected.empty() && options.explicit_selection) {
    result.diagnostics = BuildDiagnostics(options);
  }
  return result;
}

std::vector<int> ParseIndexList(const std::string& text) {
  std::vector<int> out;
  std::string token;
  auto flush = [&]() {
    if (token.empty()) return;
    size_t dash = token.find('-');
    if (dash == std::string::npos) {
      try {
        int value = std::stoi(token);
        if (value >= 0) out.push_back(value);
      } catch (...) {
      }
    } else {
      try {
        int start = std::stoi(token.substr(0, dash));
        int end = std::stoi(token.substr(dash + 1));
        if (start > end) std::swap(start, end);
        for (int v = start; v <= end; ++v) {
          if (v >= 0) out.push_back(v);
        }
      } catch (...) {
      }
    }
    token.clear();
  };
  for (char c : text) {
    if (c == ',' || c == ';') {
      flush();
    } else if (!std::isspace(static_cast<unsigned char>(c))) {
      token.push_back(c);
    }
  }
  flush();
  return out;
}

}  // namespace lattice::runtime
