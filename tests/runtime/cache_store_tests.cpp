#include "test_util.h"

#include <chrono>
#include <filesystem>
#include <sstream>

#include "runtime/backends/cache_store.h"

namespace test {

namespace {

std::filesystem::path MakeTempDir() {
  const auto base = std::filesystem::temp_directory_path();
  const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::ostringstream name;
  name << "lattice_cache_test_" << now;
  std::filesystem::path path = base / name.str();
  std::filesystem::create_directories(path);
  return path;
}

}  // namespace

void RunCacheStoreTests(TestContext* ctx) {
  const std::filesystem::path root = MakeTempDir();

  rt::CachePolicy policy;
  policy.max_bytes = 1024;
  policy.max_entries = 16;
  policy.max_age_seconds = 0;
  policy.enabled = true;
  policy.update_atime = true;

  rt::CacheStore store("opencl", policy, root);
  rt::CacheKey key{"key1", "fp1"};
  std::string error;
  store.WriteBinary(key, "abc", 3, &error);
  std::string out;
  bool hit = store.ReadBinary(key, &out, &error);
  ExpectTrue(hit && out == "abc", "cache_store_read_write", ctx);

  rt::CacheKey mismatch{"key1", "fp2"};
  std::string out_miss;
  bool miss = store.ReadBinary(mismatch, &out_miss, &error);
  ExpectTrue(!miss, "cache_store_fingerprint_miss", ctx);

  rt::CachePolicy small_policy = policy;
  small_policy.max_bytes = 5;
  rt::CacheStore evict("cuda", small_policy, root);
  rt::CacheKey evict_a{"a", "fp1"};
  rt::CacheKey evict_b{"b", "fp1"};
  evict.WriteBinary(evict_a, "aaa", 3, &error);
  evict.WriteBinary(evict_b, "bbb", 3, &error);
  std::string out_a;
  std::string out_b;
  bool hit_a = evict.ReadBinary(evict_a, &out_a, &error);
  bool hit_b = evict.ReadBinary(evict_b, &out_b, &error);
  ExpectTrue(hit_a != hit_b, "cache_store_eviction", ctx);

  rt::DeviceMetadata meta;
  meta.backend = "opencl";
  meta.index = 0;
  meta.name = "Test Device";
  meta.vendor = "Vendor";
  meta.driver_version = "1.0";
  meta.runtime_version = "1.0";
  meta.device_version = "1.2";
  meta.platform_name = "Test Platform";
  meta.platform_vendor = "Vendor";
  meta.platform_version = "1.2";
  meta.total_mem = 1024;
  meta.multiprocessor_count = 4;
  meta.clock_khz = 1000;
  meta.is_gpu = true;
  meta.fp16 = 1;
  meta.fp64 = 0;

  rt::DeviceMetadataStore meta_store(root);
  std::string meta_error;
  bool wrote = meta_store.Write(meta, &meta_error);
  const std::string fingerprint = rt::DeviceFingerprint(meta);
  rt::DeviceMetadata loaded;
  bool read = meta_store.Read(fingerprint, &loaded, &meta_error);
  ExpectTrue(wrote && read, "device_meta_roundtrip", ctx);
  ExpectTrue(loaded.name == meta.name && loaded.vendor == meta.vendor, "device_meta_fields", ctx);

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

}  // namespace test
