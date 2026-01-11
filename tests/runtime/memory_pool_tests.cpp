#include "runtime/backends/memory_pool.h"
#include "test_util.h"

#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace test {

namespace {

struct FakeAllocator {
  std::unordered_map<void*, size_t> allocations;
  size_t alloc_calls = 0;
  size_t free_calls = 0;
  size_t scrub_calls = 0;

  rt::StatusOr<rt::PoolBlock> Allocate(size_t bytes, size_t alignment) {
    ++alloc_calls;
    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(bytes, alignment);
    if (!ptr) {
      return rt::Status::Internal("fake alloc failed");
    }
#else
    if (posix_memalign(&ptr, alignment, bytes) != 0 || !ptr) {
      return rt::Status::Internal("fake alloc failed");
    }
#endif
    allocations[ptr] = bytes;
    rt::PoolBlock block;
    block.key = reinterpret_cast<uintptr_t>(ptr);
    block.handle = reinterpret_cast<uintptr_t>(ptr);
    block.host_ptr = ptr;
    block.bytes = bytes;
    block.alignment = alignment;
    return block;
  }

  rt::Status Free(const rt::PoolBlock& block) {
    ++free_calls;
    void* ptr = reinterpret_cast<void*>(block.handle);
    if (!ptr) return rt::Status::OK();
    auto it = allocations.find(ptr);
    if (it == allocations.end()) {
      return rt::Status::Invalid("unknown allocation");
    }
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
    allocations.erase(it);
    return rt::Status::OK();
  }

  rt::Status Scrub(const rt::PoolBlock& block) {
    ++scrub_calls;
    if (block.host_ptr && block.bytes > 0) {
      std::memset(block.host_ptr, 0, block.bytes);
    }
    return rt::Status::OK();
  }
};

}  // namespace

void RunMemoryPoolTests(TestContext* ctx) {
  {
    FakeAllocator alloc;
    rt::MemoryPoolConfig config = rt::DefaultDevicePoolConfig();
    config.max_pool_bytes = 1024;
    config.max_pool_entries = 8;
    config.max_entry_bytes = 512;
    config.bucket_bytes = 64;
    rt::MemoryPool pool(
        "pool_basic", config,
        [&alloc](size_t bytes, size_t alignment) { return alloc.Allocate(bytes, alignment); },
        [&alloc](const rt::PoolBlock& block) { return alloc.Free(block); },
        [&alloc](const rt::PoolBlock& block) { return alloc.Scrub(block); });

    auto a_or = pool.Acquire(96, 64);
    ExpectTrue(a_or.ok(), "pool_alloc_status", ctx);
    auto a = a_or.value();
    ExpectTrue(!a.from_pool, "pool_alloc_not_from_pool", ctx);
    ExpectTrue(pool.Release(a.key).ok(), "pool_release_status", ctx);
    auto b_or = pool.Acquire(96, 64);
    ExpectTrue(b_or.ok(), "pool_reuse_status", ctx);
    auto b = b_or.value();
    ExpectTrue(b.from_pool, "pool_reuse_from_pool", ctx);
    ExpectTrue(pool.Release(b.key).ok(), "pool_reuse_release", ctx);

    auto stats = pool.Stats();
    ExpectTrue(stats.pool_hits == 1, "pool_hit_count", ctx);
    ExpectTrue(stats.pool_misses >= 1, "pool_miss_count", ctx);
  }

  {
    FakeAllocator alloc;
    rt::MemoryPoolConfig config = rt::DefaultPinnedPoolConfig();
    config.scrub_on_free = true;
    config.max_pool_bytes = 1024;
    config.max_pool_entries = 4;
    config.max_entry_bytes = 512;
    rt::MemoryPool pool(
        "pool_scrub", config,
        [&alloc](size_t bytes, size_t alignment) { return alloc.Allocate(bytes, alignment); },
        [&alloc](const rt::PoolBlock& block) { return alloc.Free(block); },
        [&alloc](const rt::PoolBlock& block) { return alloc.Scrub(block); });

    auto block_or = pool.Acquire(128, 64);
    ExpectTrue(block_or.ok(), "pool_scrub_alloc_status", ctx);
    auto block = block_or.value();
    std::memset(block.host_ptr, 0xAB, block.bytes);
    ExpectTrue(pool.Release(block.key).ok(), "pool_scrub_release", ctx);
    auto reuse_or = pool.Acquire(128, 64);
    ExpectTrue(reuse_or.ok(), "pool_scrub_reuse_status", ctx);
    auto reuse = reuse_or.value();
    bool zeroed = true;
    auto* bytes = static_cast<unsigned char*>(reuse.host_ptr);
    for (size_t i = 0; i < reuse.bytes; ++i) {
      if (bytes[i] != 0) {
        zeroed = false;
        break;
      }
    }
    ExpectTrue(zeroed, "pool_scrub_zeroed", ctx);
    ExpectTrue(pool.Release(reuse.key).ok(), "pool_scrub_release_reuse", ctx);
  }

  {
    FakeAllocator alloc;
    rt::MemoryPoolConfig config = rt::DefaultDevicePoolConfig();
    config.max_pool_bytes = 64;
    config.max_pool_entries = 1;
    config.max_entry_bytes = 256;
    rt::MemoryPool pool(
        "pool_eviction", config,
        [&alloc](size_t bytes, size_t alignment) { return alloc.Allocate(bytes, alignment); },
        [&alloc](const rt::PoolBlock& block) { return alloc.Free(block); },
        [&alloc](const rt::PoolBlock& block) { return alloc.Scrub(block); });

    auto block_or = pool.Acquire(128, 64);
    ExpectTrue(block_or.ok(), "pool_evict_alloc_status", ctx);
    auto block = block_or.value();
    ExpectTrue(pool.Release(block.key).ok(), "pool_evict_release", ctx);
    auto stats = pool.Stats();
    ExpectTrue(stats.evictions >= 1, "pool_evict_count", ctx);
    ExpectTrue(stats.cached_blocks == 0, "pool_evict_cache_empty", ctx);
  }
}

}  // namespace test
