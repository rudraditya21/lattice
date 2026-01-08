#ifndef LATTICE_RUNTIME_BACKENDS_DEVICE_CAPS_H_
#define LATTICE_RUNTIME_BACKENDS_DEVICE_CAPS_H_

#include <cstddef>
#include <cstdint>

#include "runtime/backends/device_quirks.h"

namespace lattice::runtime {

enum class CapabilityStatus : uint8_t { kUnknown, kNo, kYes };

struct DeviceCapabilities {
  CapabilityStatus fp16 = CapabilityStatus::kUnknown;
  CapabilityStatus fp64 = CapabilityStatus::kUnknown;
  bool is_cpu = false;
  bool is_gpu = false;
  bool is_software = false;
  size_t local_mem_bytes = 0;
  size_t max_work_group_size = 0;
  size_t max_work_item_sizes[3] = {0, 0, 0};
  size_t max_threads_per_block = 0;
  size_t shared_mem_bytes = 0;
  DeviceQuirkInfo quirks;
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_DEVICE_CAPS_H_
