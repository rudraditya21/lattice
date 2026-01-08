#ifndef LATTICE_RUNTIME_BACKENDS_DEVICE_QUIRKS_H_
#define LATTICE_RUNTIME_BACKENDS_DEVICE_QUIRKS_H_

#include <cstdint>
#include <string>

#include "runtime/backend.h"

namespace lattice::runtime {

enum DeviceQuirkFlag : uint32_t {
  kNoQuirk = 0,
  kSoftwareEmulation = 1u << 0,
  kDisableFp16 = 1u << 1,
  kDisableFp64 = 1u << 2,
  kPrefer1DLaunch = 1u << 3,
};

struct DeviceQuirkInfo {
  uint32_t flags = 0;
  bool disabled = false;
  std::string reason;
};

DeviceQuirkInfo QueryDeviceQuirks(BackendType backend, const std::string& vendor,
                                  const std::string& name, const std::string& driver);

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_DEVICE_QUIRKS_H_
