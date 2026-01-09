#ifndef LATTICE_RUNTIME_BACKENDS_BACKEND_ERROR_H_
#define LATTICE_RUNTIME_BACKENDS_BACKEND_ERROR_H_

#include <string>

#include "runtime/backend.h"

namespace lattice::runtime {

enum class BackendErrorKind {
  kUnknown,
  kInit,
  kDiscovery,
  kContext,
  kMemory,
  kBuild,
  kCompile,
  kLaunch,
  kIo,
  kUnsupported,
  kInvalidArgument,
  kRuntime,
};

const char* BackendTypeName(BackendType backend);
const char* BackendErrorKindName(BackendErrorKind kind);
std::string FormatBackendErrorPrefix(BackendType backend, BackendErrorKind kind);
Status MakeBackendError(StatusCode code, BackendType backend, BackendErrorKind kind,
                        const std::string& message);

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_BACKENDS_BACKEND_ERROR_H_
