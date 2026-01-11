#include "runtime/backends/backend_log.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>

#include "runtime/backends/cache_store.h"

namespace lattice::runtime {

namespace {

struct LogConfig {
  bool enabled = false;
  LogLevel level = LogLevel::kError;
  LogFormat format = LogFormat::kText;
};

bool IsTrueEnv(const char* value) {
  if (!value) return false;
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

LogLevel ParseLogLevel(const char* value) {
  if (!value) return LogLevel::kError;
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (v == "error") return LogLevel::kError;
  if (v == "warn" || v == "warning") return LogLevel::kWarn;
  if (v == "info") return LogLevel::kInfo;
  if (v == "debug") return LogLevel::kDebug;
  if (v == "trace") return LogLevel::kTrace;
  return LogLevel::kError;
}

LogFormat ParseLogFormat(const char* value) {
  if (!value) return LogFormat::kText;
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (v == "json") return LogFormat::kJson;
  return LogFormat::kText;
}

LogConfig LoadLogConfig() {
  LogConfig config;
  if (const char* level = std::getenv("LATTICE_LOG_LEVEL")) {
    config.enabled = true;
    config.level = ParseLogLevel(level);
  }
  if (const char* fmt = std::getenv("LATTICE_LOG_FORMAT")) {
    config.enabled = true;
    config.format = ParseLogFormat(fmt);
  }
  if (const char* enable = std::getenv("LATTICE_LOG")) {
    if (IsTrueEnv(enable)) {
      config.enabled = true;
      config.level = LogLevel::kInfo;
    }
  }
  return config;
}

bool BackendVerboseEnabledImpl(BackendType backend) {
  const char* env = nullptr;
  switch (backend) {
    case BackendType::kOpenCL:
      env = std::getenv("LATTICE_OPENCL_VERBOSE");
      break;
    case BackendType::kCUDA:
      env = std::getenv("LATTICE_CUDA_VERBOSE");
      break;
    case BackendType::kHIP:
      env = std::getenv("LATTICE_HIP_VERBOSE");
      break;
    case BackendType::kMetal:
      env = std::getenv("LATTICE_METAL_VERBOSE");
      break;
    case BackendType::kCPU:
      env = std::getenv("LATTICE_CPU_VERBOSE");
      break;
  }
  return env && env[0] != '\0';
}

bool ShouldLog(const LogConfig& config, const LogRecord& record) {
  if (config.enabled) {
    return static_cast<int>(record.level) <= static_cast<int>(config.level);
  }
  return BackendVerboseEnabledImpl(record.backend);
}

std::string JsonEscape(const std::string& input) {
  std::string out;
  out.reserve(input.size() + 8);
  for (char c : input) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

std::string TextEscape(const std::string& input) {
  std::string out;
  out.reserve(input.size() + 8);
  for (char c : input) {
    if (c == '"') {
      out += "\\\"";
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else if (c == '\t') {
      out += "\\t";
    } else {
      out.push_back(c);
    }
  }
  return out;
}

const char* LogLevelName(LogLevel level) {
  switch (level) {
    case LogLevel::kError:
      return "error";
    case LogLevel::kWarn:
      return "warn";
    case LogLevel::kInfo:
      return "info";
    case LogLevel::kDebug:
      return "debug";
    case LogLevel::kTrace:
      return "trace";
  }
  return "info";
}

std::string SanitizeName(const std::string& name) {
  std::string out;
  out.reserve(name.size());
  for (char c : name) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  if (out.empty()) out = "kernel";
  return out;
}

std::filesystem::path TraceRoot() {
  if (const char* env = std::getenv("LATTICE_TRACE_DIR")) {
    return std::filesystem::path(env);
  }
  return DefaultCacheRoot() / "trace";
}

bool TraceEnabled() {
  return IsTrueEnv(std::getenv("LATTICE_TRACE_KERNELS"));
}

std::mutex g_log_mu;
std::atomic<uint64_t> g_trace_counter{0};

}  // namespace

bool BackendVerboseEnabled(BackendType backend) {
  return BackendVerboseEnabledImpl(backend);
}

std::string FormatLogLine(const LogRecord& record, LogFormat format) {
  if (format == LogFormat::kJson) {
    std::ostringstream out;
    out << "{";
    out << "\"level\":\"" << LogLevelName(record.level) << "\"";
    out << ",\"backend\":\"" << BackendTypeName(record.backend) << "\"";
    out << ",\"kind\":\"" << BackendErrorKindName(record.kind) << "\"";
    if (!record.operation.empty()) {
      out << ",\"op\":\"" << JsonEscape(record.operation) << "\"";
    }
    if (record.device_index >= 0) {
      out << ",\"device_index\":" << record.device_index;
    }
    if (!record.device_name.empty()) {
      out << ",\"device_name\":\"" << JsonEscape(record.device_name) << "\"";
    }
    if (record.error_code != 0) {
      out << ",\"error_code\":" << record.error_code;
    }
    if (!record.error_name.empty()) {
      out << ",\"error_name\":\"" << JsonEscape(record.error_name) << "\"";
    }
    if (!record.trace_path.empty()) {
      out << ",\"trace_path\":\"" << JsonEscape(record.trace_path) << "\"";
    }
    out << ",\"message\":\"" << JsonEscape(record.message) << "\"";
    out << "}";
    return out.str();
  }

  std::ostringstream out;
  out << "level=" << LogLevelName(record.level);
  out << " backend=" << BackendTypeName(record.backend);
  out << " kind=" << BackendErrorKindName(record.kind);
  if (!record.operation.empty()) out << " op=" << record.operation;
  if (record.device_index >= 0) out << " device_index=" << record.device_index;
  if (!record.device_name.empty())
    out << " device_name=\"" << TextEscape(record.device_name) << "\"";
  if (record.error_code != 0) out << " error_code=" << record.error_code;
  if (!record.error_name.empty()) out << " error_name=" << record.error_name;
  if (!record.trace_path.empty()) out << " trace_path=" << record.trace_path;
  out << " message=\"" << TextEscape(record.message) << "\"";
  return out.str();
}

void LogBackend(const LogRecord& record) {
  const LogConfig config = LoadLogConfig();
  if (!ShouldLog(config, record)) return;
  const std::string line = FormatLogLine(record, config.format);
  std::lock_guard<std::mutex> lock(g_log_mu);
  std::cerr << line << "\n";
}

bool TraceKernelSource(const KernelTrace& trace, std::string* out_path) {
  if (!TraceEnabled()) return false;
  std::filesystem::path root = TraceRoot();
  const std::filesystem::path backend_dir = root / BackendTypeName(trace.backend);
  std::error_code ec;
  std::filesystem::create_directories(backend_dir, ec);
  if (ec) return false;
  const uint64_t counter = g_trace_counter.fetch_add(1, std::memory_order_relaxed);
  const std::string name = SanitizeName(trace.kernel_name);
  const std::string filename = name + "_" + std::to_string(counter) + ".trace";
  const std::filesystem::path path = backend_dir / filename;
  std::ofstream out(path);
  if (!out) return false;
  out << "# backend=" << BackendTypeName(trace.backend) << "\n";
  if (trace.device_index >= 0) out << "# device_index=" << trace.device_index << "\n";
  if (!trace.device_name.empty()) out << "# device_name=" << trace.device_name << "\n";
  if (!trace.kernel_name.empty()) out << "# kernel=" << trace.kernel_name << "\n";
  if (!trace.build_options.empty()) out << "# build_options=" << trace.build_options << "\n";
  out << trace.source;
  if (out_path) *out_path = path.string();
  return true;
}

}  // namespace lattice::runtime
