// Entry point for the lattice REPL or script runner.
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include "builtin/builtins.h"
#include "repl/repl.h"
#include "runtime/backend.h"
#include "runtime/backends/cuda_backend.h"
#include "runtime/backends/hip_backend.h"
#include "runtime/backends/opencl_backend.h"
#if defined(__APPLE__)
#include "runtime/backends/metal_backend.h"
#endif
#include "runtime/runner.h"
#include "util/error.h"

namespace {

std::string SelectedDeviceLabel(const lattice::runtime::Backend* backend) {
  using lattice::runtime::BackendType;
  if (!backend) return "";
  switch (backend->Type()) {
    case BackendType::kOpenCL: {
      auto* ocl = static_cast<const lattice::runtime::OpenCLBackend*>(backend);
      auto info = ocl->DeviceInfo();
      if (!info.empty()) {
        return "device " + std::to_string(info[0].index) + ": " + info[0].name;
      }
      break;
    }
    case BackendType::kCUDA: {
      auto* cuda = static_cast<const lattice::runtime::CudaBackend*>(backend);
      auto info = cuda->DeviceInfo();
      if (!info.empty()) {
        return "device " + std::to_string(info[0].index) + ": " + info[0].name;
      }
      break;
    }
    case BackendType::kHIP: {
      auto* hip = static_cast<const lattice::runtime::HipBackend*>(backend);
      auto info = hip->DeviceInfo();
      if (!info.empty()) {
        return "device " + std::to_string(info[0].index) + ": " + info[0].name;
      }
      break;
    }
#if defined(__APPLE__)
    case BackendType::kMetal: {
      auto* metal = static_cast<const lattice::runtime::MetalBackend*>(backend);
      auto info = metal->DeviceInfo();
      if (!info.empty()) {
        return "device " + std::to_string(info[0].index) + ": " + info[0].name;
      }
      break;
    }
#endif
    case BackendType::kCPU:
      break;
  }
  return "";
}

void PrintBackendSelection() {
  const auto* backend = lattice::runtime::GetDefaultBackend();
  if (!backend) return;
  std::ostringstream out;
  out << "Using backend: " << backend->Name();
  const std::string device = SelectedDeviceLabel(backend);
  if (!device.empty()) {
    out << " (" << device << ")";
  }
  std::cerr << out.str() << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  // Simple CLI: no args -> REPL; arg[1] -> run script file.
  if (argc > 1) {
    std::ifstream in(argv[1]);
    if (!in) {
      std::cerr << "Could not open file: " << argv[1] << "\n";
      return 1;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    PrintBackendSelection();
    auto env = std::make_shared<lattice::runtime::Environment>();
    lattice::builtin::InstallBuiltins(env);
    lattice::builtin::InstallPrint(env);
    try {
      lattice::runtime::ExecResult result =
          lattice::runtime::RunSource(buffer.str(), env);
      (void)result;  // Script mode only prints via explicit print().
      return 0;
    } catch (const lattice::util::Error& err) {
      std::cerr << err.formatted() << "\n";
      return 1;
    } catch (const std::exception& ex) {
      std::cerr << "Unhandled error: " << ex.what() << "\n";
      return 1;
    }
  } else {
    PrintBackendSelection();
    lattice::repl::Repl repl;
    repl.Run();
    return 0;
  }
}
