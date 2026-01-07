#include "runtime/backends/metal_backend.h"

namespace lattice::runtime {

MetalBackend::MetalBackend() = default;

MetalBackend::~MetalBackend() = default;

BackendType MetalBackend::Type() const {
  return BackendType::kMetal;
}

std::string MetalBackend::Name() const {
  return "Metal";
}

BackendCapabilities MetalBackend::Capabilities() const {
  BackendCapabilities caps;
  caps.supports_dense = true;
  caps.supported_dtypes = {DType::kF32, DType::kF64, DType::kI32, DType::kU32};
  return caps;
}

StatusOr<std::shared_ptr<Stream>> MetalBackend::CreateStream() const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

StatusOr<std::shared_ptr<Event>> MetalBackend::CreateEvent() const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

StatusOr<Allocation> MetalBackend::Allocate(size_t, size_t) const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

Status MetalBackend::Deallocate(const Allocation&) const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

int MetalBackend::NumThreads() const {
  return 1;
}

size_t MetalBackend::OutstandingAllocs() const {
  return 0;
}

void MetalBackend::SetDefaultPriority(int) {}

void MetalBackend::SetDeterministic(bool) {}

int MetalBackend::DeviceCount() const {
  return 0;
}

std::vector<MetalDeviceDesc> MetalBackend::DeviceInfo() const {
  return {};
}

StatusOr<MetalBuffer> MetalBackend::CreateBuffer(int, size_t) const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

Status MetalBackend::ReleaseBuffer(MetalBuffer*) const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

Status MetalBackend::WriteBuffer(int, const MetalBuffer&, const void*, size_t, size_t) const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

Status MetalBackend::ReadBuffer(int, const MetalBuffer&, void*, size_t, size_t) const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

StatusOr<MetalKernel> MetalBackend::BuildKernelFromFile(const std::string&, const std::string&,
                                                        const std::string&) const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

StatusOr<std::vector<MetalKernel>> MetalBackend::BuildKernelsFromFile(const std::string&,
                                                                      const std::string&,
                                                                      const std::string&) const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

Status MetalBackend::ReleaseKernel(MetalKernel*) const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

Status MetalBackend::LaunchKernel(const MetalKernel&, const MetalLaunchConfig&,
                                  const std::vector<MetalKernelArg>&) const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

Status MetalBackend::SmokeTest() const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

Status MetalBackend::EnsureInitialized() const {
  return Status::Unavailable("Metal backend not supported on this platform");
}

std::string MetalBackend::KernelDir() const {
  return "";
}

std::string MetalBackend::CacheKey(const DeviceContext&, const std::string&, const std::string&,
                                   const std::string&) const {
  return "";
}

const Backend* GetMetalBackend() {
  static MetalBackend* backend = [] { return new MetalBackend(); }();
  return backend;
}

Status RunMetalSmokeTest() {
  return Status::Unavailable("Metal backend not supported on this platform");
}

}  // namespace lattice::runtime
