#include "runtime/backends/backend_error.h"

namespace lattice::runtime {

const char* BackendTypeName(BackendType backend) {
  switch (backend) {
    case BackendType::kCPU:
      return "cpu";
    case BackendType::kOpenCL:
      return "opencl";
    case BackendType::kCUDA:
      return "cuda";
    case BackendType::kHIP:
      return "hip";
    case BackendType::kMetal:
      return "metal";
  }
  return "unknown";
}

const char* BackendErrorKindName(BackendErrorKind kind) {
  switch (kind) {
    case BackendErrorKind::kInit:
      return "init";
    case BackendErrorKind::kDiscovery:
      return "discovery";
    case BackendErrorKind::kContext:
      return "context";
    case BackendErrorKind::kMemory:
      return "memory";
    case BackendErrorKind::kBuild:
      return "build";
    case BackendErrorKind::kCompile:
      return "compile";
    case BackendErrorKind::kLaunch:
      return "launch";
    case BackendErrorKind::kIo:
      return "io";
    case BackendErrorKind::kUnsupported:
      return "unsupported";
    case BackendErrorKind::kInvalidArgument:
      return "invalid_argument";
    case BackendErrorKind::kRuntime:
      return "runtime";
    case BackendErrorKind::kUnknown:
    default:
      return "unknown";
  }
}

std::string FormatBackendErrorPrefix(BackendType backend, BackendErrorKind kind) {
  std::string prefix;
  prefix.reserve(32);
  prefix.push_back('[');
  prefix.append(BackendTypeName(backend));
  prefix.append("][");
  prefix.append(BackendErrorKindName(kind));
  prefix.append("] ");
  return prefix;
}

Status MakeBackendError(StatusCode code, BackendType backend, BackendErrorKind kind,
                        const std::string& message) {
  return Status{code, FormatBackendErrorPrefix(backend, kind) + message};
}

}  // namespace lattice::runtime
