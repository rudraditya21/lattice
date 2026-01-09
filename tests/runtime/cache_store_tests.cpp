#include "test_util.h"

#include <chrono>
#include <filesystem>
#include <fstream>
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

bool ReadAccessed(const std::filesystem::path& path, const std::string& key, uint64_t* out) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  if (!std::getline(in, line)) return false;
  while (std::getline(in, line)) {
    if (line.rfind("entry", 0) != 0) continue;
    std::istringstream iss(line);
    std::string token;
    std::string found_key;
    uint64_t accessed = 0;
    while (iss >> token) {
      const size_t eq = token.find('=');
      if (eq == std::string::npos) continue;
      const std::string name = token.substr(0, eq);
      const std::string value = token.substr(eq + 1);
      if (name == "key") found_key = value;
      if (name == "accessed") accessed = std::strtoull(value.c_str(), nullptr, 10);
    }
    if (found_key == key) {
      if (out) *out = accessed;
      return true;
    }
  }
  return false;
}

void WriteIndexFile(const std::filesystem::path& path, const std::string& backend,
                    const std::string& key, const std::string& fingerprint, uint64_t accessed,
                    uint64_t created = 0, uint64_t size = 3) {
  std::ofstream out(path);
  out << "lattice-cache-index v1 backend=" << backend << "\n";
  out << "entry key=" << key << " fingerprint=" << fingerprint << " size=" << size
      << " created=" << created << " accessed=" << accessed << "\n";
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

  rt::CachePolicy env_policy = rt::LoadCachePolicyFromEnv();
  ExpectTrue(env_policy.enabled, "cache_policy_default_enabled", ctx);

  {
    ScopedEnvVar max_bytes("LATTICE_CACHE_MAX_BYTES", "128K");
    ScopedEnvVar max_entries("LATTICE_CACHE_MAX_ENTRIES", "12");
    ScopedEnvVar max_age("LATTICE_CACHE_MAX_AGE_DAYS", "2");
    ScopedEnvVar update_atime("LATTICE_CACHE_UPDATE_ATIME", "0");
    auto policy_env = rt::LoadCachePolicyFromEnv();
    ExpectTrue(policy_env.max_bytes == 131072, "cache_policy_max_bytes", ctx);
    ExpectTrue(policy_env.max_entries == 12, "cache_policy_max_entries", ctx);
    ExpectTrue(policy_env.max_age_seconds == 2ull * 24ull * 60ull * 60ull,
               "cache_policy_max_age", ctx);
    ExpectTrue(!policy_env.update_atime, "cache_policy_update_atime", ctx);
  }

  {
    ScopedEnvVar disable("LATTICE_CACHE_DISABLE", "1");
    auto policy_env = rt::LoadCachePolicyFromEnv();
    ExpectTrue(!policy_env.enabled, "cache_policy_disable", ctx);
  }

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

  {
    rt::CachePolicy no_atime = policy;
    no_atime.update_atime = false;
    rt::CacheStore atime("hip", no_atime, root);
    rt::CacheKey atime_key{"atime", "fp1"};
    atime.WriteBinary(atime_key, "data", 4, &error);
    const auto index_path = root / "hip" / "index.txt";
    uint64_t before = 0;
    bool read_before = ReadAccessed(index_path, "atime", &before);
    std::string out_value;
    atime.ReadBinary(atime_key, &out_value, &error);
    uint64_t after = 0;
    bool read_after = ReadAccessed(index_path, "atime", &after);
    ExpectTrue(read_before && read_after && before == after, "cache_no_atime_update", ctx);
  }

  {
    const std::filesystem::path backend_dir = root / "broken";
    std::filesystem::create_directories(backend_dir);
    std::ofstream out(backend_dir / "index.txt");
    out << "lattice-cache-index v0 backend=broken\n";
    out << "entry key=bad fingerprint=fp size=3 created=0 accessed=0\n";
    out.close();
    std::ofstream bin(backend_dir / "bad.bin", std::ios::binary);
    bin << "abc";
    bin.close();
    rt::CacheStore bad_store("broken", policy, root);
    std::string bad_out;
    bool bad_hit = bad_store.ReadBinary({"bad", "fp"}, &bad_out, &error);
    ExpectTrue(!bad_hit, "cache_index_version_mismatch", ctx);
    ExpectTrue(!std::filesystem::exists(backend_dir / "bad.bin"), "cache_index_reset", ctx);
  }

  {
    const std::filesystem::path backend_dir = root / "cuda";
    std::filesystem::create_directories(backend_dir);
    WriteIndexFile(backend_dir / "index.txt", "cuda", "old", "fp", 0);
    std::ofstream bin(backend_dir / "old.bin", std::ios::binary);
    bin << "abc";
    bin.close();
    rt::CachePolicy age_policy = policy;
    age_policy.max_age_seconds = 1;
    rt::CacheStore age_store("cuda", age_policy, root);
    age_store.Prune();
    ExpectTrue(!std::filesystem::exists(backend_dir / "old.bin"), "cache_age_eviction", ctx);
  }

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
