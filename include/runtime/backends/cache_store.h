#ifndef LATTICE_RUNTIME_BACKENDS_CACHE_STORE_H_
#define LATTICE_RUNTIME_BACKENDS_CACHE_STORE_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>

namespace lattice::runtime {

struct CachePolicy {
  uint64_t max_bytes = 512ull * 1024ull * 1024ull;
  uint64_t max_entries = 4096;
  uint64_t max_age_seconds = 30ull * 24ull * 60ull * 60ull;
  bool enabled = true;
  bool update_atime = true;
};

CachePolicy LoadCachePolicyFromEnv();
std::filesystem::path DefaultCacheRoot();

struct CacheKey {
  std::string key;
  std::string fingerprint;
};

class CacheStore {
 public:
  CacheStore(const std::string& backend, CachePolicy policy = LoadCachePolicyFromEnv(),
             std::filesystem::path root = DefaultCacheRoot());

  bool Enabled() const { return policy_.enabled; }
  bool ReadBinary(const CacheKey& key, std::string* out, std::string* error);
  bool WriteBinary(const CacheKey& key, const void* data, size_t size, std::string* error);
  void Prune();

 private:
  struct Entry {
    std::string key;
    std::string fingerprint;
    uint64_t size = 0;
    uint64_t created = 0;
    uint64_t accessed = 0;
  };

  void EnsureLoaded();
  bool LoadIndex(std::string* error);
  bool FlushIndex(std::string* error);
  void ResetCache();
  void EvictIfNeeded();
  std::filesystem::path EntryPath(const std::string& key) const;
  void RemoveEntry(const std::string& key);

  std::string backend_;
  CachePolicy policy_;
  std::filesystem::path root_;
  std::filesystem::path backend_dir_;
  std::filesystem::path index_path_;
  bool loaded_ = false;
  std::unordered_map<std::string, Entry> entries_;
  std::mutex mu_;
};

struct DeviceMetadata {
  std::string backend;
  std::string name;
  std::string vendor;
  std::string driver_version;
  std::string runtime_version;
  std::string device_version;
  std::string platform_name;
  std::string platform_vendor;
  std::string platform_version;
  int index = -1;
  uint64_t total_mem = 0;
  int multiprocessor_count = 0;
  int clock_khz = 0;
  bool is_cpu = false;
  bool is_gpu = false;
  bool is_accel = false;
  int fp16 = -1;
  int fp64 = -1;
};

std::string DeviceFingerprint(const DeviceMetadata& meta);

class DeviceMetadataStore {
 public:
  explicit DeviceMetadataStore(std::filesystem::path root = DefaultCacheRoot());

  bool Write(const DeviceMetadata& meta, std::string* error);
  bool Read(const std::string& fingerprint, DeviceMetadata* meta, std::string* error);

 private:
  std::filesystem::path MetaPath(const std::string& fingerprint) const;
  bool EnsureDir(std::string* error);

  std::filesystem::path root_;
  std::filesystem::path device_dir_;
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_CACHE_STORE_H_
