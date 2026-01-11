#ifndef LATTICE_RUNTIME_BACKENDS_BACKEND_LOG_H_
#define LATTICE_RUNTIME_BACKENDS_BACKEND_LOG_H_

#include <cstdint>
#include <filesystem>
#include <string>

#include "runtime/backends/backend_error.h"

namespace lattice::runtime {

enum class LogLevel { kError, kWarn, kInfo, kDebug, kTrace };
enum class LogFormat { kText, kJson };

struct LogRecord {
  LogLevel level = LogLevel::kInfo;
  BackendType backend = BackendType::kCPU;
  BackendErrorKind kind = BackendErrorKind::kUnknown;
  std::string message;
  std::string operation;
  int device_index = -1;
  std::string device_name;
  int64_t error_code = 0;
  std::string error_name;
  std::string trace_path;
};

struct KernelTrace {
  BackendType backend = BackendType::kCPU;
  std::string kernel_name;
  std::string build_options;
  std::string source;
  int device_index = -1;
  std::string device_name;
};

std::string FormatLogLine(const LogRecord& record, LogFormat format);
void LogBackend(const LogRecord& record);
bool BackendVerboseEnabled(BackendType backend);
bool TraceKernelSource(const KernelTrace& trace, std::string* out_path);

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_BACKEND_LOG_H_
