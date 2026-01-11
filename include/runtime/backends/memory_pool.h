#ifndef LATTICE_RUNTIME_BACKENDS_MEMORY_POOL_H_
#define LATTICE_RUNTIME_BACKENDS_MEMORY_POOL_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/backend.h"

namespace lattice::runtime {

struct MemoryPoolConfig {
  bool enabled = true;
  bool scrub_on_free = false;
  bool scrub_on_alloc = false;
  size_t max_pool_bytes = 256 * 1024 * 1024;
  size_t max_pool_entries = 4096;
  size_t max_entry_bytes = 64 * 1024 * 1024;
  size_t bucket_bytes = 256;
};

MemoryPoolConfig DefaultDevicePoolConfig();
MemoryPoolConfig DefaultPinnedPoolConfig();
MemoryPoolConfig LoadMemoryPoolConfig(const std::string& prefix, MemoryPoolConfig base);
void AccumulateMemoryPoolStats(MemoryPoolStats* dst, const MemoryPoolStats& src);

struct PoolBlock {
  uintptr_t key = 0;
  uintptr_t handle = 0;
  void* host_ptr = nullptr;
  size_t bytes = 0;
  size_t alignment = 0;
  bool from_pool = false;
};

class MemoryPool {
 public:
  using AllocFn = std::function<StatusOr<PoolBlock>(size_t bytes, size_t alignment)>;
  using FreeFn = std::function<Status(const PoolBlock&)>;
  using ScrubFn = std::function<Status(const PoolBlock&)>;

  MemoryPool(std::string label, MemoryPoolConfig config, AllocFn alloc_fn, FreeFn free_fn,
             ScrubFn scrub_fn);

  StatusOr<PoolBlock> Acquire(size_t bytes, size_t alignment);
  Status Release(uintptr_t key);
  Status Trim();
  MemoryPoolStats Stats() const;
  size_t Outstanding() const;
  MemoryPoolConfig Config() const;

 private:
  struct ActiveInfo {
    PoolBlock block;
    size_t requested = 0;
  };

  size_t BucketFor(size_t bytes, size_t alignment) const;
  bool ShouldPool(size_t bytes) const;
  void UpdatePeakLocked();

  std::string label_;
  MemoryPoolConfig config_;
  AllocFn alloc_fn_;
  FreeFn free_fn_;
  ScrubFn scrub_fn_;
  mutable std::mutex mu_;
  std::unordered_map<uintptr_t, ActiveInfo> active_;
  std::unordered_map<size_t, std::vector<PoolBlock>> cached_;
  MemoryPoolStats stats_;
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_MEMORY_POOL_H_
