#ifndef LATTICE_RUNTIME_BACKENDS_CUDA_BACKEND_H_
#define LATTICE_RUNTIME_BACKENDS_CUDA_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/backend.h"
#include "runtime/backends/gpu/cuda_loader.h"

namespace lattice::runtime {

struct CudaDeviceDesc {
  std::string name;
  int major = 0;
  int minor = 0;
  size_t total_mem = 0;
  int multiprocessor_count = 0;
  int clock_khz = 0;
};

struct CudaBuffer {
  gpu::CUdeviceptr ptr = 0;
  size_t bytes = 0;
};

struct CudaKernel {
  gpu::CUmodule module = nullptr;
  gpu::CUfunction func = nullptr;
  int device_index = -1;
  std::string name;
};

struct CudaKernelArg {
  enum class Kind { kDevicePtr, kValue };
  Kind kind = Kind::kValue;
  gpu::CUdeviceptr device_ptr = 0;
  const void* value = nullptr;
  size_t size = 0;

  static CudaKernelArg Device(gpu::CUdeviceptr ptr) {
    CudaKernelArg arg;
    arg.kind = Kind::kDevicePtr;
    arg.device_ptr = ptr;
    return arg;
  }

  static CudaKernelArg Value(const void* data, size_t data_size) {
    CudaKernelArg arg;
    arg.kind = Kind::kValue;
    arg.value = data;
    arg.size = data_size;
    return arg;
  }
};

struct CudaLaunchConfig {
  uint32_t grid[3] = {1, 1, 1};
  uint32_t block[3] = {1, 1, 1};
  size_t shared_bytes = 0;
};

class CudaBackend final : public Backend {
 public:
  CudaBackend();
  ~CudaBackend() override;

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
  std::vector<CudaDeviceDesc> DeviceInfo() const;

  StatusOr<CudaBuffer> CreateBuffer(int device_index, size_t bytes) const;
  Status ReleaseBuffer(CudaBuffer* buffer) const;
  Status WriteBuffer(int device_index, const CudaBuffer& buffer, const void* data, size_t bytes,
                     size_t offset = 0) const;
  Status ReadBuffer(int device_index, const CudaBuffer& buffer, void* data, size_t bytes,
                    size_t offset = 0) const;

  StatusOr<CudaKernel> BuildKernelFromFile(const std::string& path, const std::string& kernel_name,
                                           const std::string& extra_build_options = "") const;
  StatusOr<std::vector<CudaKernel>> BuildKernelsFromFile(
      const std::string& path, const std::string& kernel_name,
      const std::string& extra_build_options = "") const;
  Status ReleaseKernel(CudaKernel* kernel) const;
  Status LaunchKernel(const CudaKernel& kernel, const CudaLaunchConfig& config,
                      const std::vector<CudaKernelArg>& args) const;

  Status SmokeTest() const;

 private:
  struct DeviceContext;

  Status EnsureInitialized() const;
  std::string KernelDir() const;
  std::string BuildOptions(const DeviceContext& dev, const std::string& extra) const;
  std::string CacheKey(const DeviceContext& dev, const std::string& kernel_name,
                       const std::string& build_options, const std::string& source) const;
  StatusOr<gpu::CUmodule> BuildOrLoadModule(DeviceContext& dev, const std::string& source,
                                            const std::string& build_options,
                                            const std::string& cache_key,
                                            bool source_is_binary) const;

  mutable std::vector<DeviceContext> devices_;
  mutable std::unordered_map<gpu::CUdeviceptr, size_t> allocations_;
  mutable std::mutex alloc_mu_;
  mutable gpu::CudaLoader loader_;
  mutable bool initialized_ = false;
  mutable Status init_status_ = Status::OK();
  mutable std::mutex mu_;
  int default_priority_ = 0;
  bool deterministic_ = false;
};

const Backend* GetCudaBackend();
Status RunCudaSmokeTest();

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_CUDA_BACKEND_H_
