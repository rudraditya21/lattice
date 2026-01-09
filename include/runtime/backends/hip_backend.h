#ifndef LATTICE_RUNTIME_BACKENDS_HIP_BACKEND_H_
#define LATTICE_RUNTIME_BACKENDS_HIP_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/backend.h"
#include "runtime/backends/device_caps.h"
#include "runtime/backends/gpu/hip_loader.h"

namespace lattice::runtime {

struct HipDeviceDesc {
  int index = -1;
  std::string name;
  std::string vendor;
  std::string driver_version;
  std::string runtime_version;
  size_t total_mem = 0;
  int multiprocessor_count = 0;
  int clock_khz = 0;
};

struct HipBuffer {
  gpu::hipDeviceptr_t ptr = nullptr;
  size_t bytes = 0;
};

struct HipKernel {
  gpu::hipModule_t module = nullptr;
  gpu::hipFunction_t func = nullptr;
  int device_index = -1;
  std::string name;
};

struct HipKernelArg {
  enum class Kind { kDevicePtr, kValue };
  Kind kind = Kind::kValue;
  gpu::hipDeviceptr_t device_ptr = nullptr;
  const void* value = nullptr;
  size_t size = 0;

  static HipKernelArg Device(gpu::hipDeviceptr_t ptr) {
    HipKernelArg arg;
    arg.kind = Kind::kDevicePtr;
    arg.device_ptr = ptr;
    return arg;
  }

  static HipKernelArg Value(const void* data, size_t data_size) {
    HipKernelArg arg;
    arg.kind = Kind::kValue;
    arg.value = data;
    arg.size = data_size;
    return arg;
  }
};

struct HipLaunchConfig {
  uint32_t grid[3] = {1, 1, 1};
  uint32_t block[3] = {1, 1, 1};
  size_t shared_bytes = 0;
};

class HipBackend final : public Backend {
 public:
  HipBackend();
  ~HipBackend() override;

  BackendType Type() const override;
  std::string Name() const override;
  BackendCapabilities Capabilities() const override;
  StatusOr<std::shared_ptr<Stream>> CreateStream() const override;
  StatusOr<std::shared_ptr<Event>> CreateEvent() const override;
  StatusOr<Allocation> Allocate(size_t bytes, size_t alignment = 64) const override;
  Status Deallocate(const Allocation& alloc) const override;
  int NumThreads() const override;
  size_t OutstandingAllocs() const override;
  void SetDefaultPriority(int priority) override;
  void SetDeterministic(bool deterministic) override;

  int DeviceCount() const;
  std::vector<HipDeviceDesc> DeviceInfo() const;
  std::vector<DeviceCapabilities> DeviceCaps() const;

  StatusOr<HipBuffer> CreateBuffer(int device_index, size_t bytes) const;
  Status ReleaseBuffer(HipBuffer* buffer) const;
  Status WriteBuffer(int device_index, const HipBuffer& buffer, const void* data, size_t bytes,
                     size_t offset = 0) const;
  Status ReadBuffer(int device_index, const HipBuffer& buffer, void* data, size_t bytes,
                    size_t offset = 0) const;

  StatusOr<HipKernel> BuildKernelFromFile(const std::string& path, const std::string& kernel_name,
                                          const std::string& extra_build_options = "") const;
  StatusOr<std::vector<HipKernel>> BuildKernelsFromFile(
      const std::string& path, const std::string& kernel_name,
      const std::string& extra_build_options = "") const;
  Status ReleaseKernel(HipKernel* kernel) const;
  Status LaunchKernel(const HipKernel& kernel, const HipLaunchConfig& config,
                      const std::vector<HipKernelArg>& args) const;

  Status SmokeTest() const;

 private:
  struct DeviceContext;

  Status EnsureInitialized() const;
  std::string KernelDir() const;
  std::string BuildOptions(const DeviceContext& dev, const std::string& extra) const;
  std::string CacheKey(const DeviceContext& dev, const std::string& kernel_name,
                       const std::string& build_options, const std::string& source) const;
  StatusOr<gpu::hipModule_t> BuildOrLoadModule(DeviceContext& dev, const std::string& source,
                                               const std::string& build_options,
                                               const std::string& cache_key, bool source_is_binary,
                                               const std::string& kernel_name) const;

  mutable std::vector<DeviceContext> devices_;
  mutable std::unordered_map<gpu::hipDeviceptr_t, size_t> allocations_;
  mutable std::mutex alloc_mu_;
  mutable gpu::HipLoader loader_;
  mutable bool initialized_ = false;
  mutable Status init_status_ = Status::OK();
  mutable std::mutex mu_;
  int default_priority_ = 0;
  bool deterministic_ = false;
};

const Backend* GetHipBackend();
Status RunHipSmokeTest();

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_HIP_BACKEND_H_
