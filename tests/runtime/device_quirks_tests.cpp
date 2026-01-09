#include "test_util.h"

#include "runtime/backends/device_quirks.h"

namespace test {

void RunDeviceQuirksTests(TestContext* ctx) {
  {
    ScopedEnvVar ignore("LATTICE_IGNORE_DEVICE_QUIRKS", "1");
    auto info =
        rt::QueryDeviceQuirks(rt::BackendType::kOpenCL, "Portable Computing Language", "pocl", "");
    ExpectTrue(info.flags == 0 && !info.disabled, "device_quirks_ignore_table", ctx);
  }

  {
    ScopedEnvVar blacklist("LATTICE_DEVICE_BLACKLIST", "nvidia");
    auto info = rt::QueryDeviceQuirks(rt::BackendType::kCUDA, "NVIDIA", "GeForce RTX", "driver");
    ExpectTrue(info.disabled, "device_quirks_blacklist", ctx);
    ExpectTrue(info.reason.find("blacklisted") != std::string::npos, "device_quirks_reason", ctx);
  }

  {
    ScopedEnvVar disable_sw("LATTICE_DISABLE_SOFTWARE_DEVICES", "1");
    auto info =
        rt::QueryDeviceQuirks(rt::BackendType::kOpenCL, "Portable Computing Language", "pocl", "");
    ExpectTrue((info.flags & rt::kSoftwareEmulation) != 0, "device_quirks_sw_flag", ctx);
    ExpectTrue(info.disabled, "device_quirks_sw_disabled", ctx);
  }
}

}  // namespace test
