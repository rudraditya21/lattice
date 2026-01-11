#include "runtime/backends/gpu/hip_loader.h"

#include <cstring>

namespace lattice::runtime::gpu {

namespace {

template <typename T>
bool LoadSymbol(DynLib lib, const char* name, T* out, std::string* error) {
  auto sym = DlSym(lib, name);
  if (!sym) {
    if (error) {
      *error = std::string("Missing HIP symbol: ") + name;
    }
    return false;
  }
  *out = reinterpret_cast<T>(sym);
  return true;
}

template <typename T>
bool LoadOptional(DynLib lib, const char* name, T* out) {
  auto sym = DlSym(lib, name);
  if (!sym) return false;
  *out = reinterpret_cast<T>(sym);
  return true;
}

bool LoadHipLib(DynLib* lib, std::string* error) {
#if defined(_WIN32)
  const char* libs[] = {"amdhip64.dll", "hip64.dll"};
#elif defined(__APPLE__)
  const char* libs[] = {"libamdhip64.dylib", "libhip64.dylib"};
#else
  const char* libs[] = {"libamdhip64.so", "libamdhip64.so.5", "libhip64.so"};
#endif
  for (const char* name : libs) {
    *lib = DlOpen(name);
    if (*lib) return true;
  }
  if (error) {
    *error = "Failed to load HIP runtime: " + DlError();
  }
  return false;
}

bool LoadHiprtcLib(DynLib* lib) {
#if defined(_WIN32)
  const char* libs[] = {"hiprtc.dll"};
#elif defined(__APPLE__)
  const char* libs[] = {"libhiprtc.dylib"};
#else
  const char* libs[] = {"libhiprtc.so", "libhiprtc.so.5"};
#endif
  for (const char* name : libs) {
    *lib = DlOpen(name);
    if (*lib) return true;
  }
  return false;
}

}  // namespace

bool HipLoader::Load(std::string* error) {
  if (driver) return true;

  if (!LoadHipLib(&driver, error)) {
    return false;
  }

  bool ok = true;
  LoadOptional(driver, "hipInit", &hipInit);
  LoadOptional(driver, "hipDriverGetVersion", &hipDriverGetVersion);
  ok &= LoadSymbol(driver, "hipGetDeviceCount", &hipGetDeviceCount, error);
  ok &= LoadSymbol(driver, "hipDeviceGet", &hipDeviceGet, error);
  ok &= LoadSymbol(driver, "hipDeviceGetName", &hipDeviceGetName, error);
  LoadOptional(driver, "hipDeviceTotalMem", &hipDeviceTotalMem);
  LoadOptional(driver, "hipDeviceGetAttribute", &hipDeviceGetAttribute);
  LoadOptional(driver, "hipCtxCreate", &hipCtxCreate);
  LoadOptional(driver, "hipCtxDestroy", &hipCtxDestroy);
  LoadOptional(driver, "hipCtxSetCurrent", &hipCtxSetCurrent);
  ok &= LoadSymbol(driver, "hipStreamCreate", &hipStreamCreate, error);
  ok &= LoadSymbol(driver, "hipStreamDestroy", &hipStreamDestroy, error);
  LoadOptional(driver, "hipStreamSynchronize", &hipStreamSynchronize);
  ok &= LoadSymbol(driver, "hipMalloc", &hipMalloc, error);
  ok &= LoadSymbol(driver, "hipFree", &hipFree, error);
  LoadOptional(driver, "hipHostMalloc", &hipHostMalloc);
  LoadOptional(driver, "hipHostFree", &hipHostFree);
  LoadOptional(driver, "hipMemset", &hipMemset);
  ok &= LoadSymbol(driver, "hipMemcpy", &hipMemcpy, error);
  LoadOptional(driver, "hipModuleLoadData", &hipModuleLoadData);
  LoadOptional(driver, "hipModuleUnload", &hipModuleUnload);
  LoadOptional(driver, "hipModuleGetFunction", &hipModuleGetFunction);
  LoadOptional(driver, "hipModuleLaunchKernel", &hipModuleLaunchKernel);
  LoadOptional(driver, "hipGetErrorString", &hipGetErrorString);
  LoadOptional(driver, "hipRuntimeGetVersion", &hipRuntimeGetVersion);

  if (!ok) {
    Unload();
    return false;
  }

  if (LoadHiprtcLib(&hiprtc)) {
    LoadSymbol(hiprtc, "hiprtcCreateProgram", &hiprtcCreateProgram, nullptr);
    LoadSymbol(hiprtc, "hiprtcCompileProgram", &hiprtcCompileProgram, nullptr);
    LoadSymbol(hiprtc, "hiprtcDestroyProgram", &hiprtcDestroyProgram, nullptr);
    LoadSymbol(hiprtc, "hiprtcGetCodeSize", &hiprtcGetCodeSize, nullptr);
    LoadSymbol(hiprtc, "hiprtcGetCode", &hiprtcGetCode, nullptr);
    LoadSymbol(hiprtc, "hiprtcGetProgramLogSize", &hiprtcGetProgramLogSize, nullptr);
    LoadSymbol(hiprtc, "hiprtcGetProgramLog", &hiprtcGetProgramLog, nullptr);
    LoadSymbol(hiprtc, "hiprtcGetErrorString", &hiprtcGetErrorString, nullptr);
  }

  return true;
}

void HipLoader::Unload() {
  if (driver) {
    DlClose(driver);
    driver = nullptr;
  }
  if (hiprtc) {
    DlClose(hiprtc);
    hiprtc = nullptr;
  }
}

std::string HipErrorString(hipError_t err, const HipLoader* loader) {
  if (loader && loader->hipGetErrorString) {
    const char* msg = loader->hipGetErrorString(err);
    if (msg) return msg;
  }
  return "HIP error " + std::to_string(err);
}

std::string HiprtcErrorString(HiprtcResult err, const HipLoader* loader) {
  if (loader && loader->hiprtcGetErrorString) {
    const char* msg = loader->hiprtcGetErrorString(err);
    if (msg) return msg;
  }
  return "HIPRTC error " + std::to_string(err);
}

}  // namespace lattice::runtime::gpu
