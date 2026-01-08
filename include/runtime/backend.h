#ifndef LATTICE_RUNTIME_BACKEND_H_
#define LATTICE_RUNTIME_BACKEND_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "runtime/dtype.h"

namespace lattice::runtime {

enum class BackendType { kCPU, kOpenCL, kCUDA, kHIP, kMetal };

enum class StatusCode { kOk, kInvalidArgument, kUnavailable, kInternal };

struct Status {
  StatusCode code = StatusCode::kOk;
  std::string message;
  static Status OK() { return Status{StatusCode::kOk, ""}; }
  static Status Invalid(const std::string& msg) {
    return Status{StatusCode::kInvalidArgument, msg};
  }
  static Status Unavailable(const std::string& msg) {
    return Status{StatusCode::kUnavailable, msg};
  }
  static Status Internal(const std::string& msg) { return Status{StatusCode::kInternal, msg}; }
  bool ok() const { return code == StatusCode::kOk; }
};

template <typename T>
class StatusOr {
 public:
  StatusOr(const Status& s) : status_(s), has_value_(false) {}
  StatusOr(T v) : status_(Status::OK()), value_(std::move(v)), has_value_(true) {}
  template <typename U, typename = std::enable_if_t<std::is_convertible<U, T>::value>>
  StatusOr(U&& v) : status_(Status::OK()), value_(std::forward<U>(v)), has_value_(true) {}
  const Status& status() const { return status_; }
  bool ok() const { return status_.ok(); }
  const T& value() const { return value_; }
  T& value() { return value_; }

 private:
  Status status_;
  T value_{};
  bool has_value_ = false;
};

struct BackendCapabilities {
  bool supports_dense = true;
  bool supports_sparse = false;
  bool supports_ragged = false;
  bool supports_fft = false;
  bool supports_blas = false;
  bool supports_conv = false;
  bool supports_rng = true;
  bool supports_events = true;
  std::vector<DType> supported_dtypes;
};

struct Allocation {
  void* ptr = nullptr;
  void* device_handle = nullptr;
  size_t bytes = 0;
  size_t alignment = 64;  // default cacheline alignment
  bool from_pool = false;
  int numa_node = -1;
};

class Event {
 public:
  virtual ~Event() = default;
  virtual void Record() = 0;
  virtual void Wait() = 0;
  virtual bool Ready() const = 0;
};

class Stream {
 public:
  virtual ~Stream() = default;
  virtual void Submit(std::function<void()> fn) = 0;
  virtual void Synchronize() = 0;
  virtual void AddDependency(const std::shared_ptr<Event>& ev) = 0;
  virtual void SetPriority(int priority) = 0;  // higher means higher priority
};

struct TensorDesc {
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DType dtype = DType::kF64;
  bool row_major = true;
};

struct BlasMatDesc {
  TensorDesc tensor;
  bool transpose = false;
};

struct Conv2DDesc {
  TensorDesc input;
  TensorDesc filter;
  std::vector<int64_t> padding{0, 0};
  std::vector<int64_t> stride{1, 1};
};

struct FFTDesc {
  TensorDesc tensor;
  bool inverse = false;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual BackendType Type() const = 0;
  virtual std::string Name() const = 0;
  virtual BackendCapabilities Capabilities() const = 0;
  virtual StatusOr<std::shared_ptr<Stream>> CreateStream() const = 0;
  virtual StatusOr<std::shared_ptr<Event>> CreateEvent() const = 0;
  virtual StatusOr<Allocation> Allocate(size_t bytes, size_t alignment = 64) const = 0;
  virtual Status Deallocate(const Allocation& alloc) const = 0;
  virtual int NumThreads() const = 0;
  virtual size_t OutstandingAllocs() const = 0;
  virtual void SetDefaultPriority(int priority) = 0;
  virtual void SetDeterministic(bool deterministic) = 0;
};

class CpuBackend final : public Backend {
 public:
  CpuBackend();
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

 private:
  int preferred_numa_node_ = -1;  // -1 means default OS policy
  int default_priority_ = 0;
};

// Returns a singleton CPU backend instance.
const Backend* GetCpuBackend();
const Backend* GetOpenCLBackend();
const Backend* GetCudaBackend();
const Backend* GetHipBackend();
const Backend* GetBackendByType(BackendType type);
// Returns the default backend (CPU today).
const Backend* GetDefaultBackend();
Status RunOpenCLSmokeTest();
Status RunCudaSmokeTest();
Status RunHipSmokeTest();
#if defined(__APPLE__)
const Backend* GetMetalBackend();
Status RunMetalSmokeTest();
#endif

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKEND_H_
