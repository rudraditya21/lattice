#ifndef LATTICE_RUNTIME_BACKENDS_OPENCL_BACKEND_H_
#define LATTICE_RUNTIME_BACKENDS_OPENCL_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/backend.h"
#include "runtime/backends/device_caps.h"
#include "runtime/backends/gpu/opencl_loader.h"

namespace lattice::runtime {

struct OpenCLDeviceDesc {
  int index = -1;
  std::string name;
  std::string vendor;
  std::string platform_name;
  std::string platform_vendor;
  std::string platform_version;
  std::string device_version;
  std::string runtime_version;
  std::string driver_version;
  cl_device_type type = 0;
  cl_uint vendor_id = 0;
  size_t max_work_group_size = 0;
  cl_ulong local_mem_size = 0;
  cl_device_local_mem_type local_mem_type = CL_LOCAL;
  cl_uint compute_units = 0;
  cl_uint max_clock_mhz = 0;
};

struct OpenCLBuffer {
  cl_mem mem = nullptr;
  size_t bytes = 0;
};

struct OpenCLKernel {
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  int device_index = -1;
  std::string name;
};

struct OpenCLKernelArg {
  enum class Kind { kMem, kValue };
  Kind kind = Kind::kValue;
  cl_mem mem = nullptr;
  const void* value = nullptr;
  size_t size = 0;

  static OpenCLKernelArg Mem(cl_mem mem_handle) {
    OpenCLKernelArg arg;
    arg.kind = Kind::kMem;
    arg.mem = mem_handle;
    return arg;
  }

  static OpenCLKernelArg Value(const void* data, size_t data_size) {
    OpenCLKernelArg arg;
    arg.kind = Kind::kValue;
    arg.value = data;
    arg.size = data_size;
    return arg;
  }
};

struct OpenCLLaunchConfig {
  cl_uint dims = 1;
  size_t global[3] = {0, 0, 0};
  size_t local[3] = {0, 0, 0};
  bool use_local = false;
};

class OpenCLBackend final : public Backend {
 public:
  OpenCLBackend();
  ~OpenCLBackend() override;

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
  std::vector<OpenCLDeviceDesc> DeviceInfo() const;
  std::vector<DeviceCapabilities> DeviceCaps() const;

  StatusOr<OpenCLBuffer> CreateBuffer(int device_index, size_t bytes,
                                      cl_mem_flags flags = CL_MEM_READ_WRITE) const;
  Status ReleaseBuffer(OpenCLBuffer* buffer) const;
  Status WriteBuffer(int device_index, const OpenCLBuffer& buffer, const void* data, size_t bytes,
                     size_t offset = 0) const;
  Status ReadBuffer(int device_index, const OpenCLBuffer& buffer, void* data, size_t bytes,
                    size_t offset = 0) const;

  StatusOr<OpenCLKernel> BuildKernelFromFile(const std::string& path,
                                             const std::string& kernel_name,
                                             const std::string& extra_build_options = "") const;
  StatusOr<std::vector<OpenCLKernel>> BuildKernelsFromFile(
      const std::string& path, const std::string& kernel_name,
      const std::string& extra_build_options = "") const;
  Status ReleaseKernel(OpenCLKernel* kernel) const;
  Status LaunchKernel(const OpenCLKernel& kernel, const OpenCLLaunchConfig& config,
                      const std::vector<OpenCLKernelArg>& args) const;

  Status SmokeTest() const;

 private:
  struct DeviceContext;

  Status EnsureInitialized() const;
  std::string KernelDir() const;
  std::string DeviceInfoString(cl_device_id device, cl_device_info param) const;
  std::string PlatformInfoString(cl_platform_id platform, cl_platform_info param) const;
  std::string BuildOptions(const DeviceContext& dev, const std::string& extra) const;
  int DeviceIndex(const DeviceContext& dev) const;
  std::string CacheKey(const DeviceContext& dev, const std::string& kernel_name,
                       const std::string& build_options, const std::string& source) const;
  StatusOr<cl_program> BuildOrLoadProgram(DeviceContext& dev, const std::string& source,
                                          const std::string& build_options,
                                          const std::string& cache_key) const;

  mutable std::vector<DeviceContext> devices_;
  mutable std::unordered_map<cl_mem, size_t> allocations_;
  mutable std::mutex alloc_mu_;
  mutable gpu::OpenCLLoader loader_;
  mutable bool initialized_ = false;
  mutable Status init_status_ = Status::OK();
  mutable std::mutex mu_;
  int default_priority_ = 0;
  bool deterministic_ = false;
};

const Backend* GetOpenCLBackend();
Status RunOpenCLSmokeTest();

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_OPENCL_BACKEND_H_
