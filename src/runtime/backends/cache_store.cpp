#include "runtime/backends/cache_store.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

namespace lattice::runtime {

namespace {

constexpr int kCacheIndexVersion = 1;
constexpr int kDeviceMetaVersion = 1;

uint64_t NowSeconds() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(now).count());
}

bool IsTrueEnvValue(const char* value) {
  if (!value) return false;
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

bool ParseUint64(const char* value, uint64_t* out) {
  if (!value || !out) return false;
  char* end = nullptr;
  uint64_t parsed = std::strtoull(value, &end, 10);
  if (end == value) return false;
  *out = parsed;
  return true;
}

int ParseIntValue(const std::string& value, int fallback) {
  char* end = nullptr;
  long parsed = std::strtol(value.c_str(), &end, 10);
  if (end == value.c_str()) return fallback;
  return static_cast<int>(parsed);
}

uint64_t ParseUint64Value(const std::string& value, uint64_t fallback) {
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
  if (end == value.c_str()) return fallback;
  return static_cast<uint64_t>(parsed);
}

bool ParseSize(const char* value, uint64_t* out) {
  if (!value || !out) return false;
  std::string v(value);
  if (v.empty()) return false;
  char suffix = static_cast<char>(std::toupper(v.back()));
  uint64_t multiplier = 1;
  if (suffix == 'K' || suffix == 'M' || suffix == 'G') {
    v.pop_back();
    if (suffix == 'K') multiplier = 1024ull;
    if (suffix == 'M') multiplier = 1024ull * 1024ull;
    if (suffix == 'G') multiplier = 1024ull * 1024ull * 1024ull;
  }
  char* end = nullptr;
  uint64_t parsed = std::strtoull(v.c_str(), &end, 10);
  if (end == v.c_str()) return false;
  *out = parsed * multiplier;
  return true;
}

uint64_t Fnv1a64(const std::string& data) {
  uint64_t hash = 14695981039346656037ull;
  for (unsigned char c : data) {
    hash ^= static_cast<uint64_t>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string Hex64(uint64_t value) {
  static const char* kHex = "0123456789abcdef";
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i) {
    out[static_cast<size_t>(i)] = kHex[value & 0xF];
    value >>= 4;
  }
  return out;
}

std::vector<std::string> SplitTokens(const std::string& line) {
  std::vector<std::string> tokens;
  std::string current;
  std::istringstream in(line);
  while (in >> current) {
    tokens.push_back(current);
  }
  return tokens;
}

std::unordered_map<std::string, std::string> ParseKeyValueTokens(const std::string& line) {
  std::unordered_map<std::string, std::string> out;
  const auto tokens = SplitTokens(line);
  for (const auto& token : tokens) {
    const size_t eq = token.find('=');
    if (eq == std::string::npos) continue;
    out[token.substr(0, eq)] = token.substr(eq + 1);
  }
  return out;
}

bool ReadFile(const std::filesystem::path& path, std::string* out, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error) *error = "Failed to open file: " + path.string();
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  *out = ss.str();
  return true;
}

bool WriteFileAtomically(const std::filesystem::path& path, const void* data, size_t size,
                         std::string* error) {
  std::filesystem::path tmp = path;
  tmp += ".tmp";
  std::ofstream out(tmp, std::ios::binary);
  if (!out) {
    if (error) *error = "Failed to open file for write: " + tmp.string();
    return false;
  }
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!out) {
    if (error) *error = "Failed to write file: " + tmp.string();
    return false;
  }
  out.close();
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    if (error) *error = "Failed to rename cache file: " + ec.message();
    return false;
  }
  return true;
}

}  // namespace

CachePolicy LoadCachePolicyFromEnv() {
  CachePolicy policy;
  if (IsTrueEnvValue(std::getenv("LATTICE_CACHE_DISABLE"))) {
    policy.enabled = false;
    return policy;
  }
  if (const char* env = std::getenv("LATTICE_CACHE_MAX_BYTES")) {
    uint64_t value = 0;
    if (ParseSize(env, &value)) policy.max_bytes = value;
  }
  if (const char* env = std::getenv("LATTICE_CACHE_MAX_ENTRIES")) {
    uint64_t value = 0;
    if (ParseUint64(env, &value)) policy.max_entries = value;
  }
  if (const char* env = std::getenv("LATTICE_CACHE_MAX_AGE_DAYS")) {
    uint64_t value = 0;
    if (ParseUint64(env, &value)) policy.max_age_seconds = value * 24ull * 60ull * 60ull;
  }
  if (const char* env = std::getenv("LATTICE_CACHE_UPDATE_ATIME")) {
    policy.update_atime = IsTrueEnvValue(env);
  }
  return policy;
}

std::filesystem::path DefaultCacheRoot() {
  if (const char* env = std::getenv("LATTICE_CACHE_DIR")) {
    return std::filesystem::path(env);
  }
  return std::filesystem::current_path() / ".lattice_cache";
}

CacheStore::CacheStore(const std::string& backend, CachePolicy policy, std::filesystem::path root)
    : backend_(backend),
      policy_(policy),
      root_(std::move(root)),
      backend_dir_(root_ / backend_),
      index_path_(backend_dir_ / "index.txt") {}

bool CacheStore::ReadBinary(const CacheKey& key, std::string* out, std::string* error) {
  if (!policy_.enabled) return false;
  if (!out) return false;
  std::lock_guard<std::mutex> lock(mu_);
  EnsureLoaded();
  auto it = entries_.find(key.key);
  if (it == entries_.end()) return false;
  if (!key.fingerprint.empty() && it->second.fingerprint != key.fingerprint) {
    RemoveEntry(key.key);
    FlushIndex(nullptr);
    return false;
  }
  const std::filesystem::path path = EntryPath(key.key);
  if (!std::filesystem::exists(path)) {
    RemoveEntry(key.key);
    FlushIndex(nullptr);
    return false;
  }
  if (!ReadFile(path, out, error)) {
    RemoveEntry(key.key);
    FlushIndex(nullptr);
    return false;
  }
  if (policy_.update_atime) {
    it->second.accessed = NowSeconds();
    FlushIndex(nullptr);
  }
  return true;
}

bool CacheStore::WriteBinary(const CacheKey& key, const void* data, size_t size,
                             std::string* error) {
  if (!policy_.enabled) return false;
  std::lock_guard<std::mutex> lock(mu_);
  EnsureLoaded();
  std::error_code ec;
  std::filesystem::create_directories(backend_dir_, ec);
  if (ec) {
    if (error) *error = "Failed to create cache dir: " + ec.message();
    return false;
  }
  const std::filesystem::path path = EntryPath(key.key);
  if (!WriteFileAtomically(path, data, size, error)) {
    return false;
  }
  Entry entry;
  entry.key = key.key;
  entry.fingerprint = key.fingerprint;
  entry.size = static_cast<uint64_t>(size);
  entry.created = NowSeconds();
  entry.accessed = entry.created;
  entries_[key.key] = entry;
  EvictIfNeeded();
  FlushIndex(nullptr);
  return true;
}

void CacheStore::Prune() {
  if (!policy_.enabled) return;
  std::lock_guard<std::mutex> lock(mu_);
  EnsureLoaded();
  EvictIfNeeded();
  FlushIndex(nullptr);
}

void CacheStore::EnsureLoaded() {
  if (loaded_) return;
  std::string error;
  if (!LoadIndex(&error)) {
    ResetCache();
  }
  loaded_ = true;
}

bool CacheStore::LoadIndex(std::string* error) {
  entries_.clear();
  if (!std::filesystem::exists(index_path_)) {
    return true;
  }
  std::ifstream in(index_path_);
  if (!in) {
    if (error) *error = "Failed to open cache index: " + index_path_.string();
    return false;
  }
  std::string line;
  if (!std::getline(in, line)) return false;
  const auto header = SplitTokens(line);
  if (header.size() < 3 || header[0] != "lattice-cache-index") return false;
  const std::string expected_version = "v" + std::to_string(kCacheIndexVersion);
  if (header[1] != expected_version) return false;
  if (header[2].rfind("backend=", 0) != 0) return false;
  const std::string backend = header[2].substr(std::string("backend=").size());
  if (backend != backend_) return false;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line.rfind("entry", 0) != 0) continue;
    const auto kv = ParseKeyValueTokens(line);
    Entry entry;
    auto it = kv.find("key");
    if (it == kv.end()) continue;
    entry.key = it->second;
    it = kv.find("fingerprint");
    if (it != kv.end()) entry.fingerprint = it->second;
    it = kv.find("size");
    if (it != kv.end())
      entry.size = static_cast<uint64_t>(std::strtoull(it->second.c_str(), nullptr, 10));
    it = kv.find("created");
    if (it != kv.end())
      entry.created = static_cast<uint64_t>(std::strtoull(it->second.c_str(), nullptr, 10));
    it = kv.find("accessed");
    if (it != kv.end())
      entry.accessed = static_cast<uint64_t>(std::strtoull(it->second.c_str(), nullptr, 10));
    if (entry.key.empty()) continue;
    entries_[entry.key] = entry;
  }
  return true;
}

bool CacheStore::FlushIndex(std::string* error) {
  std::error_code ec;
  std::filesystem::create_directories(backend_dir_, ec);
  if (ec) {
    if (error) *error = "Failed to create cache dir: " + ec.message();
    return false;
  }
  std::ostringstream out;
  out << "lattice-cache-index v" << kCacheIndexVersion << " backend=" << backend_ << "\n";
  for (const auto& kv : entries_) {
    const Entry& entry = kv.second;
    out << "entry"
        << " key=" << entry.key << " fingerprint=" << entry.fingerprint << " size=" << entry.size
        << " created=" << entry.created << " accessed=" << entry.accessed << "\n";
  }
  const std::string content = out.str();
  return WriteFileAtomically(index_path_, content.data(), content.size(), error);
}

void CacheStore::ResetCache() {
  std::error_code ec;
  std::filesystem::remove_all(backend_dir_, ec);
  std::filesystem::create_directories(backend_dir_, ec);
  entries_.clear();
}

void CacheStore::EvictIfNeeded() {
  if (entries_.empty()) return;
  const uint64_t now = NowSeconds();
  std::vector<Entry> entries;
  entries.reserve(entries_.size());
  for (const auto& kv : entries_) {
    entries.push_back(kv.second);
  }

  if (policy_.max_age_seconds > 0) {
    for (const auto& entry : entries) {
      if (now - entry.accessed > policy_.max_age_seconds) {
        RemoveEntry(entry.key);
      }
    }
  }

  uint64_t total_bytes = 0;
  for (const auto& kv : entries_) {
    total_bytes += kv.second.size;
  }

  if ((policy_.max_bytes > 0 && total_bytes > policy_.max_bytes) ||
      (policy_.max_entries > 0 && entries_.size() > policy_.max_entries)) {
    std::vector<Entry> candidates;
    candidates.reserve(entries_.size());
    for (const auto& kv : entries_) {
      candidates.push_back(kv.second);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Entry& a, const Entry& b) { return a.accessed < b.accessed; });
    size_t idx = 0;
    while ((policy_.max_bytes > 0 && total_bytes > policy_.max_bytes) ||
           (policy_.max_entries > 0 && entries_.size() > policy_.max_entries)) {
      if (idx >= candidates.size()) break;
      const Entry& victim = candidates[idx++];
      total_bytes -= victim.size;
      RemoveEntry(victim.key);
    }
  }
}

std::filesystem::path CacheStore::EntryPath(const std::string& key) const {
  return backend_dir_ / (key + ".bin");
}

void CacheStore::RemoveEntry(const std::string& key) {
  auto it = entries_.find(key);
  if (it == entries_.end()) return;
  std::error_code ec;
  std::filesystem::remove(EntryPath(key), ec);
  entries_.erase(it);
}

std::string DeviceFingerprint(const DeviceMetadata& meta) {
  std::ostringstream ss;
  ss << meta.backend << "|" << meta.vendor << "|" << meta.name << "|" << meta.driver_version << "|"
     << meta.runtime_version << "|" << meta.device_version << "|" << meta.platform_name << "|"
     << meta.platform_vendor << "|" << meta.platform_version;
  return Hex64(Fnv1a64(ss.str()));
}

DeviceMetadataStore::DeviceMetadataStore(std::filesystem::path root)
    : root_(std::move(root)), device_dir_(root_ / "devices") {}

bool DeviceMetadataStore::Write(const DeviceMetadata& meta, std::string* error) {
  if (!EnsureDir(error)) return false;
  const std::string fingerprint = DeviceFingerprint(meta);
  const std::filesystem::path path = MetaPath(fingerprint);
  std::ostringstream out;
  out << "lattice-device-meta v" << kDeviceMetaVersion << "\n";
  out << "backend=" << meta.backend << "\n";
  out << "index=" << meta.index << "\n";
  out << "name=" << meta.name << "\n";
  out << "vendor=" << meta.vendor << "\n";
  out << "driver_version=" << meta.driver_version << "\n";
  out << "runtime_version=" << meta.runtime_version << "\n";
  out << "device_version=" << meta.device_version << "\n";
  out << "platform_name=" << meta.platform_name << "\n";
  out << "platform_vendor=" << meta.platform_vendor << "\n";
  out << "platform_version=" << meta.platform_version << "\n";
  out << "total_mem=" << meta.total_mem << "\n";
  out << "multiprocessor_count=" << meta.multiprocessor_count << "\n";
  out << "clock_khz=" << meta.clock_khz << "\n";
  out << "is_cpu=" << (meta.is_cpu ? 1 : 0) << "\n";
  out << "is_gpu=" << (meta.is_gpu ? 1 : 0) << "\n";
  out << "is_accel=" << (meta.is_accel ? 1 : 0) << "\n";
  out << "fp16=" << meta.fp16 << "\n";
  out << "fp64=" << meta.fp64 << "\n";
  const std::string content = out.str();
  return WriteFileAtomically(path, content.data(), content.size(), error);
}

bool DeviceMetadataStore::Read(const std::string& fingerprint, DeviceMetadata* meta,
                               std::string* error) {
  if (!meta) return false;
  const std::filesystem::path path = MetaPath(fingerprint);
  std::string content;
  if (!ReadFile(path, &content, error)) return false;
  std::istringstream in(content);
  std::string line;
  if (!std::getline(in, line)) return false;
  const std::string expected_meta = "lattice-device-meta v" + std::to_string(kDeviceMetaVersion);
  if (line != expected_meta) return false;
  DeviceMetadata out;
  while (std::getline(in, line)) {
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = line.substr(0, eq);
    const std::string value = line.substr(eq + 1);
    if (key == "backend")
      out.backend = value;
    else if (key == "index")
      out.index = ParseIntValue(value, out.index);
    else if (key == "name")
      out.name = value;
    else if (key == "vendor")
      out.vendor = value;
    else if (key == "driver_version")
      out.driver_version = value;
    else if (key == "runtime_version")
      out.runtime_version = value;
    else if (key == "device_version")
      out.device_version = value;
    else if (key == "platform_name")
      out.platform_name = value;
    else if (key == "platform_vendor")
      out.platform_vendor = value;
    else if (key == "platform_version")
      out.platform_version = value;
    else if (key == "total_mem")
      out.total_mem = ParseUint64Value(value, out.total_mem);
    else if (key == "multiprocessor_count") {
      out.multiprocessor_count = ParseIntValue(value, out.multiprocessor_count);
    } else if (key == "clock_khz") {
      out.clock_khz = ParseIntValue(value, out.clock_khz);
    } else if (key == "is_cpu") {
      out.is_cpu = value == "1";
    } else if (key == "is_gpu") {
      out.is_gpu = value == "1";
    } else if (key == "is_accel") {
      out.is_accel = value == "1";
    } else if (key == "fp16") {
      out.fp16 = ParseIntValue(value, out.fp16);
    } else if (key == "fp64") {
      out.fp64 = ParseIntValue(value, out.fp64);
    }
  }
  *meta = out;
  return true;
}

std::filesystem::path DeviceMetadataStore::MetaPath(const std::string& fingerprint) const {
  return device_dir_ / (fingerprint + ".meta");
}

bool DeviceMetadataStore::EnsureDir(std::string* error) {
  std::error_code ec;
  std::filesystem::create_directories(device_dir_, ec);
  if (ec) {
    if (error) *error = "Failed to create device meta dir: " + ec.message();
    return false;
  }
  return true;
}

}  // namespace lattice::runtime
