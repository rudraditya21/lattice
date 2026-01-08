#ifndef LATTICE_RUNTIME_BACKENDS_DEVICE_SELECTOR_H_
#define LATTICE_RUNTIME_BACKENDS_DEVICE_SELECTOR_H_

#include <string>
#include <vector>

namespace lattice::runtime {

enum class DeviceKind { kAny, kCPU, kGPU, kAccelerator };

struct DeviceIdentity {
  int index = -1;
  std::string name;
  std::string vendor;
  std::string driver;
  DeviceKind kind = DeviceKind::kAny;
};

struct DeviceSelectionOptions {
  DeviceKind kind = DeviceKind::kAny;
  std::vector<std::string> include_patterns;
  std::vector<std::string> exclude_patterns;
  std::string mask;
  std::vector<int> indices;
  std::vector<int> order;
  bool explicit_selection = false;
};

struct DeviceSelectionResult {
  std::vector<int> indices;
  std::string diagnostics;
};

DeviceSelectionOptions LoadDeviceSelectionOptions(const char* prefix);
DeviceSelectionResult SelectDevices(const std::vector<DeviceIdentity>& devices,
                                    const DeviceSelectionOptions& options);

std::vector<int> ParseIndexList(const std::string& text);

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_DEVICE_SELECTOR_H_
