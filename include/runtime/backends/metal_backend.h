#ifndef LATTICE_RUNTIME_BACKENDS_METAL_BACKEND_H_
#define LATTICE_RUNTIME_BACKENDS_METAL_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/backend.h"
#include "runtime/backends/device_caps.h"

namespace lattice::runtime {

class MemoryPool;

struct MetalDeviceDesc {
  int index = -1;
  std::string name;
  std::string vendor;
  std::string driver_version;
  std::string runtime_version;
  size_t max_threadgroup_size = 0;
  size_t shared_mem_bytes = 0;
};

struct MetalBuffer {
  void* handle = nullptr;
  size_t bytes = 0;
  int device_index = -1;
};

struct MetalKernel {
  void* pipeline = nullptr;
  int device_index = -1;
  std::string name;
};

struct MetalKernelArg {
  enum class Kind { kBuffer, kValue };
  Kind kind = Kind::kValue;
  void* buffer = nullptr;
  const void* value = nullptr;
  size_t size = 0;

  static MetalKernelArg Buffer(void* buffer_handle) {
    MetalKernelArg arg;
    arg.kind = Kind::kBuffer;
    arg.buffer = buffer_handle;
    return arg;
  }

  static MetalKernelArg Value(const void* data, size_t data_size) {
    MetalKernelArg arg;
    arg.kind = Kind::kValue;
    arg.value = data;
    arg.size = data_size;
    return arg;
  }
};

struct MetalLaunchConfig {
  uint32_t grid[3] = {1, 1, 1};
  uint32_t threads[3] = {1, 1, 1};
  bool use_threads = false;
};

class MetalBackend final : public Backend {
 public:
  MetalBackend();
  ~MetalBackend() override;

  BackendType Type() const override;
  std::string Name() const override;
  BackendCapabilities Capabilities() const override;
  StatusOr<std::shared_ptr<Stream>> CreateStream() const override;
  StatusOr<std::shared_ptr<Event>> CreateEvent() const override;
  StatusOr<Allocation> Allocate(size_t bytes, size_t alignment = 64) const override;
  Status Deallocate(const Allocation& alloc) const override;
  StatusOr<Allocation> AllocatePinned(size_t bytes, size_t alignment = 64) const override;
  Status DeallocatePinned(const Allocation& alloc) const override;
  int NumThreads() const override;
  size_t OutstandingAllocs() const override;
  BackendMemoryStats MemoryStats() const override;
  void SetDefaultPriority(int priority) override;
  void SetDeterministic(bool deterministic) override;

  int DeviceCount() const;
  std::vector<MetalDeviceDesc> DeviceInfo() const;
  std::vector<DeviceCapabilities> DeviceCaps() const;

  StatusOr<MetalBuffer> CreateBuffer(int device_index, size_t bytes) const;
  Status ReleaseBuffer(MetalBuffer* buffer) const;
  Status WriteBuffer(int device_index, const MetalBuffer& buffer, const void* data, size_t bytes,
                     size_t offset = 0) const;
  Status ReadBuffer(int device_index, const MetalBuffer& buffer, void* data, size_t bytes,
                    size_t offset = 0) const;

  StatusOr<MetalKernel> BuildKernelFromFile(const std::string& path, const std::string& kernel_name,
                                            const std::string& extra_build_options = "") const;
  StatusOr<std::vector<MetalKernel>> BuildKernelsFromFile(
      const std::string& path, const std::string& kernel_name,
      const std::string& extra_build_options = "") const;
  Status ReleaseKernel(MetalKernel* kernel) const;
  Status LaunchKernel(const MetalKernel& kernel, const MetalLaunchConfig& config,
                      const std::vector<MetalKernelArg>& args) const;

  Status SmokeTest() const;

 private:
  struct DeviceContext;

  Status EnsureInitialized() const;
  std::string KernelDir() const;
  std::string CacheKey(const DeviceContext& dev, const std::string& kernel_name,
                       const std::string& build_options, const std::string& source) const;
  MemoryPool* DevicePool(int device_index) const;
  MemoryPool* PinnedPool(int device_index) const;

  mutable std::vector<DeviceContext> devices_;
  mutable bool initialized_ = false;
  mutable Status init_status_ = Status::OK();
  mutable std::mutex mu_;
  int default_priority_ = 0;
  bool deterministic_ = false;
};

const Backend* GetMetalBackend();
Status RunMetalSmokeTest();

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_METAL_BACKEND_H_
