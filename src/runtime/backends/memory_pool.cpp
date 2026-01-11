#include "runtime/backends/memory_pool.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace lattice::runtime {

namespace {

bool IsTrueEnvValue(const char* value) {
  if (!value) return false;
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

bool ParseSizeValue(const char* value, size_t* out) {
  if (!value || !out) return false;
  std::string v(value);
  if (v.empty()) return false;
  char suffix = static_cast<char>(std::toupper(v.back()));
  size_t multiplier = 1;
  if (suffix == 'K' || suffix == 'M' || suffix == 'G') {
    v.pop_back();
    if (suffix == 'K') multiplier = 1024ull;
    if (suffix == 'M') multiplier = 1024ull * 1024ull;
    if (suffix == 'G') multiplier = 1024ull * 1024ull * 1024ull;
  }
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(v.c_str(), &end, 10);
  if (end == v.c_str()) return false;
  if (parsed > std::numeric_limits<size_t>::max() / multiplier) {
    *out = std::numeric_limits<size_t>::max();
    return true;
  }
  *out = static_cast<size_t>(parsed) * multiplier;
  return true;
}

bool ParseCountValue(const char* value, size_t* out) {
  if (!value || !out) return false;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value) return false;
  *out = static_cast<size_t>(parsed);
  return true;
}

std::string EnvKey(const std::string& prefix, const char* suffix) {
  std::ostringstream ss;
  ss << prefix;
  if (!prefix.empty() && prefix.back() != '_') {
    ss << "_";
  }
  ss << suffix;
  return ss.str();
}

size_t RoundUp(size_t value, size_t align) {
  if (align == 0) return value;
  size_t rem = value % align;
  if (rem == 0) return value;
  size_t add = align - rem;
  if (value > std::numeric_limits<size_t>::max() - add) return value;
  return value + add;
}

}  // namespace

MemoryPoolConfig DefaultDevicePoolConfig() {
  MemoryPoolConfig config;
  config.enabled = true;
  config.scrub_on_free = false;
  config.scrub_on_alloc = false;
  config.max_pool_bytes = 256ull * 1024ull * 1024ull;
  config.max_pool_entries = 4096;
  config.max_entry_bytes = 64ull * 1024ull * 1024ull;
  config.bucket_bytes = 256;
  return config;
}

MemoryPoolConfig DefaultPinnedPoolConfig() {
  MemoryPoolConfig config;
  config.enabled = true;
  config.scrub_on_free = true;
  config.scrub_on_alloc = false;
  config.max_pool_bytes = 128ull * 1024ull * 1024ull;
  config.max_pool_entries = 2048;
  config.max_entry_bytes = 32ull * 1024ull * 1024ull;
  config.bucket_bytes = 256;
  return config;
}

MemoryPoolConfig LoadMemoryPoolConfig(const std::string& prefix, MemoryPoolConfig base) {
  if (IsTrueEnvValue(std::getenv(EnvKey(prefix, "DISABLE").c_str()))) {
    base.enabled = false;
  }
  if (const char* env = std::getenv(EnvKey(prefix, "ENABLE").c_str())) {
    base.enabled = IsTrueEnvValue(env);
  }
  if (const char* env = std::getenv(EnvKey(prefix, "SCRUB").c_str())) {
    base.scrub_on_free = IsTrueEnvValue(env);
  }
  if (const char* env = std::getenv(EnvKey(prefix, "SCRUB_ON_ALLOC").c_str())) {
    base.scrub_on_alloc = IsTrueEnvValue(env);
  }
  size_t parsed = 0;
  if (ParseSizeValue(std::getenv(EnvKey(prefix, "MAX_BYTES").c_str()), &parsed)) {
    base.max_pool_bytes = parsed;
  }
  if (ParseCountValue(std::getenv(EnvKey(prefix, "MAX_ENTRIES").c_str()), &parsed)) {
    base.max_pool_entries = parsed;
  }
  if (ParseSizeValue(std::getenv(EnvKey(prefix, "MAX_ENTRY_BYTES").c_str()), &parsed)) {
    base.max_entry_bytes = parsed;
  }
  if (ParseSizeValue(std::getenv(EnvKey(prefix, "BUCKET_BYTES").c_str()), &parsed)) {
    base.bucket_bytes = std::max<size_t>(1, parsed);
  }
  return base;
}

void AccumulateMemoryPoolStats(MemoryPoolStats* dst, const MemoryPoolStats& src) {
  if (!dst) return;
  dst->in_use_bytes += src.in_use_bytes;
  dst->in_use_blocks += src.in_use_blocks;
  dst->cached_bytes += src.cached_bytes;
  dst->cached_blocks += src.cached_blocks;
  dst->total_alloc_calls += src.total_alloc_calls;
  dst->total_free_calls += src.total_free_calls;
  dst->pool_hits += src.pool_hits;
  dst->pool_misses += src.pool_misses;
  dst->evictions += src.evictions;
  dst->scrubbed_bytes += src.scrubbed_bytes;
  dst->peak_in_use_bytes = std::max(dst->peak_in_use_bytes, src.peak_in_use_bytes);
  dst->peak_in_use_blocks = std::max(dst->peak_in_use_blocks, src.peak_in_use_blocks);
}

MemoryPool::MemoryPool(std::string label, MemoryPoolConfig config, AllocFn alloc_fn, FreeFn free_fn,
                       ScrubFn scrub_fn)
    : label_(std::move(label)),
      config_(config),
      alloc_fn_(std::move(alloc_fn)),
      free_fn_(std::move(free_fn)),
      scrub_fn_(std::move(scrub_fn)) {}

StatusOr<PoolBlock> MemoryPool::Acquire(size_t bytes, size_t alignment) {
  if (bytes == 0) {
    PoolBlock empty;
    return empty;
  }
  if (alignment == 0) alignment = 1;
  PoolBlock block;
  size_t request_bytes = bytes;
  const bool use_pool = config_.enabled && ShouldPool(bytes);
  const size_t bucket = use_pool ? BucketFor(bytes, alignment) : bytes;
  {
    std::lock_guard<std::mutex> lock(mu_);
    stats_.total_alloc_calls++;
    if (use_pool) {
      auto it = cached_.find(bucket);
      if (it != cached_.end() && !it->second.empty()) {
        auto& vec = it->second;
        for (size_t i = vec.size(); i-- > 0;) {
          if (vec[i].alignment >= alignment) {
            block = vec[i];
            vec.erase(vec.begin() + static_cast<std::ptrdiff_t>(i));
            break;
          }
        }
        if (vec.empty()) cached_.erase(it);
      }
      if (block.key != 0) {
        stats_.pool_hits++;
        stats_.cached_bytes -= block.bytes;
        stats_.cached_blocks--;
      } else {
        stats_.pool_misses++;
      }
    }
  }

  if (block.key == 0) {
    auto alloc_or = alloc_fn_(bucket, alignment);
    if (!alloc_or.ok()) return alloc_or.status();
    block = alloc_or.value();
    block.from_pool = false;
  } else {
    block.from_pool = true;
  }

  if (block.key == 0) {
    return Status::Internal(label_ + " pool allocation returned null handle");
  }

  if (block.bytes == 0) {
    block.bytes = bucket;
  }

  if (config_.scrub_on_alloc && block.from_pool && scrub_fn_) {
    Status scrub_status = scrub_fn_(block);
    if (!scrub_status.ok()) {
      free_fn_(block);
      return scrub_status;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      stats_.scrubbed_bytes += block.bytes;
    }
  }

  {
    std::lock_guard<std::mutex> lock(mu_);
    ActiveInfo info;
    info.block = block;
    info.requested = request_bytes;
    active_[block.key] = info;
    stats_.in_use_bytes += block.bytes;
    stats_.in_use_blocks++;
    UpdatePeakLocked();
  }

  return block;
}

Status MemoryPool::Release(uintptr_t key) {
  if (key == 0) return Status::OK();
  ActiveInfo info;
  {
    std::lock_guard<std::mutex> lock(mu_);
    stats_.total_free_calls++;
    auto it = active_.find(key);
    if (it == active_.end()) {
      return Status::Invalid(label_ + " unknown allocation");
    }
    info = it->second;
    active_.erase(it);
    if (stats_.in_use_bytes >= info.block.bytes) {
      stats_.in_use_bytes -= info.block.bytes;
    } else {
      stats_.in_use_bytes = 0;
    }
    if (stats_.in_use_blocks > 0) {
      stats_.in_use_blocks--;
    }
  }

  if (config_.scrub_on_free && scrub_fn_) {
    Status scrub_status = scrub_fn_(info.block);
    if (!scrub_status.ok()) {
      free_fn_(info.block);
      return scrub_status;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      stats_.scrubbed_bytes += info.block.bytes;
    }
  }

  bool pooled = false;
  if (config_.enabled && ShouldPool(info.block.bytes)) {
    std::lock_guard<std::mutex> lock(mu_);
    if (stats_.cached_bytes + info.block.bytes <= config_.max_pool_bytes &&
        stats_.cached_blocks + 1 <= config_.max_pool_entries) {
      cached_[info.block.bytes].push_back(info.block);
      stats_.cached_bytes += info.block.bytes;
      stats_.cached_blocks++;
      pooled = true;
    } else {
      stats_.evictions++;
    }
  }

  if (!pooled) {
    return free_fn_(info.block);
  }
  return Status::OK();
}

Status MemoryPool::Trim() {
  std::unordered_map<size_t, std::vector<PoolBlock>> cached;
  {
    std::lock_guard<std::mutex> lock(mu_);
    cached.swap(cached_);
    stats_.cached_bytes = 0;
    stats_.cached_blocks = 0;
  }
  for (auto& bucket : cached) {
    for (const auto& block : bucket.second) {
      free_fn_(block);
    }
  }
  return Status::OK();
}

MemoryPoolStats MemoryPool::Stats() const {
  std::lock_guard<std::mutex> lock(mu_);
  return stats_;
}

size_t MemoryPool::Outstanding() const {
  std::lock_guard<std::mutex> lock(mu_);
  return active_.size();
}

MemoryPoolConfig MemoryPool::Config() const {
  return config_;
}

size_t MemoryPool::BucketFor(size_t bytes, size_t alignment) const {
  size_t bucket = bytes;
  if (config_.bucket_bytes > 0) {
    bucket = RoundUp(bucket, config_.bucket_bytes);
  }
  if (alignment > 1) {
    bucket = RoundUp(bucket, alignment);
  }
  return bucket;
}

bool MemoryPool::ShouldPool(size_t bytes) const {
  return bytes > 0 && bytes <= config_.max_entry_bytes && config_.max_pool_bytes > 0 &&
         config_.max_pool_entries > 0;
}

void MemoryPool::UpdatePeakLocked() {
  stats_.peak_in_use_bytes = std::max(stats_.peak_in_use_bytes, stats_.in_use_bytes);
  stats_.peak_in_use_blocks = std::max(stats_.peak_in_use_blocks, stats_.in_use_blocks);
}

}  // namespace lattice::runtime
