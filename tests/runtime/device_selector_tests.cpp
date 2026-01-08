#include "test_util.h"

#include "runtime/backends/device_selector.h"

namespace test {

void RunDeviceSelectorTests(TestContext* ctx) {
  const auto indices = rt::ParseIndexList("0,2-4,6");
  ExpectTrue(indices.size() == 5, "device_index_parse_len", ctx);
  ExpectTrue(indices[0] == 0 && indices[1] == 2 && indices[2] == 3 && indices[3] == 4 &&
                 indices[4] == 6,
             "device_index_parse_values", ctx);

  std::vector<rt::DeviceIdentity> devices;
  devices.push_back({0, "GeForce RTX", "NVIDIA", "driver", rt::DeviceKind::kGPU});
  devices.push_back({1, "Ryzen CPU", "AMD", "driver", rt::DeviceKind::kCPU});
  devices.push_back({2, "Radeon", "AMD", "driver", rt::DeviceKind::kGPU});

  rt::DeviceSelectionOptions opts;
  opts.kind = rt::DeviceKind::kGPU;
  opts.include_patterns = {"nvidia"};
  opts.explicit_selection = true;
  auto selected = rt::SelectDevices(devices, opts);
  ExpectTrue(selected.indices.size() == 1 && selected.indices[0] == 0,
             "device_select_vendor", ctx);

  opts = rt::DeviceSelectionOptions();
  opts.mask = "010";
  opts.explicit_selection = true;
  selected = rt::SelectDevices(devices, opts);
  ExpectTrue(selected.indices.size() == 1 && selected.indices[0] == 1,
             "device_select_mask", ctx);

  opts = rt::DeviceSelectionOptions();
  opts.indices = {2, 0};
  opts.order = {0, 2};
  opts.explicit_selection = true;
  selected = rt::SelectDevices(devices, opts);
  ExpectTrue(selected.indices.size() == 2 && selected.indices[0] == 0 && selected.indices[1] == 2,
             "device_select_order", ctx);
}

}  // namespace test
