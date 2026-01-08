#include "runtime/backends/gpu/cuda_loader.h"

#include <cstring>

namespace lattice::runtime::gpu {

namespace {

template <typename T>
bool LoadSymbol(DynLib lib, const char* name, T* out, std::string* error) {
  auto sym = DlSym(lib, name);
  if (!sym) {
    if (error) {
      *error = std::string("Missing CUDA symbol: ") + name;
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

bool LoadDriverLib(DynLib* lib, std::string* error) {
#if defined(_WIN32)
  const char* libs[] = {"nvcuda.dll"};
#elif defined(__APPLE__)
  const char* libs[] = {"libcuda.dylib", "/usr/local/cuda/lib/libcuda.dylib"};
#else
  const char* libs[] = {"libcuda.so", "libcuda.so.1"};
#endif
  for (const char* name : libs) {
    *lib = DlOpen(name);
    if (*lib) return true;
  }
  if (error) {
    *error = "Failed to load CUDA driver: " + DlError();
  }
  return false;
}

bool LoadNvrtcLib(DynLib* lib) {
#if defined(_WIN32)
  const char* libs[] = {"nvrtc64_120_0.dll", "nvrtc64_112_0.dll", "nvrtc64_111_0.dll",
                        "nvrtc64_102_0.dll", "nvrtc64_101_0.dll", "nvrtc64_100_0.dll",
                        "nvrtc64_90_0.dll",  "nvrtc64_80_0.dll",  "nvrtc64_75_0.dll",
                        "nvrtc64_70_0.dll",  "nvrtc64.dll"};
#elif defined(__APPLE__)
  const char* libs[] = {"libnvrtc.dylib", "/usr/local/cuda/lib/libnvrtc.dylib"};
#else
  const char* libs[] = {"libnvrtc.so", "libnvrtc.so.1", "libnvrtc.so.11.2", "libnvrtc.so.12",
                        "libnvrtc.so.11"};
#endif
  for (const char* name : libs) {
    *lib = DlOpen(name);
    if (*lib) return true;
  }
  return false;
}

}  // namespace

bool CudaLoader::Load(std::string* error) {
  if (driver) return true;

  if (!LoadDriverLib(&driver, error)) {
    return false;
  }

  bool ok = true;
  ok &= LoadSymbol(driver, "cuInit", &cuInit, error);
  LoadOptional(driver, "cuDriverGetVersion", &cuDriverGetVersion);
  ok &= LoadSymbol(driver, "cuDeviceGetCount", &cuDeviceGetCount, error);
  ok &= LoadSymbol(driver, "cuDeviceGet", &cuDeviceGet, error);
  ok &= LoadSymbol(driver, "cuDeviceGetName", &cuDeviceGetName, error);
  LoadOptional(driver, "cuDeviceComputeCapability", &cuDeviceComputeCapability);
  if (!LoadOptional(driver, "cuDeviceTotalMem_v2", &cuDeviceTotalMem)) {
    ok &= LoadSymbol(driver, "cuDeviceTotalMem", &cuDeviceTotalMem, error);
  }
  LoadOptional(driver, "cuDeviceGetAttribute", &cuDeviceGetAttribute);
  if (!LoadOptional(driver, "cuCtxCreate_v2", &cuCtxCreate)) {
    ok &= LoadSymbol(driver, "cuCtxCreate", &cuCtxCreate, error);
  }
  if (!LoadOptional(driver, "cuCtxDestroy_v2", &cuCtxDestroy)) {
    ok &= LoadSymbol(driver, "cuCtxDestroy", &cuCtxDestroy, error);
  }
  ok &= LoadSymbol(driver, "cuCtxSetCurrent", &cuCtxSetCurrent, error);
  if (!LoadOptional(driver, "cuStreamCreate_v2", &cuStreamCreate)) {
    ok &= LoadSymbol(driver, "cuStreamCreate", &cuStreamCreate, error);
  }
  if (!LoadOptional(driver, "cuStreamDestroy_v2", &cuStreamDestroy)) {
    ok &= LoadSymbol(driver, "cuStreamDestroy", &cuStreamDestroy, error);
  }
  LoadOptional(driver, "cuStreamSynchronize", &cuStreamSynchronize);
  if (!LoadOptional(driver, "cuMemAlloc_v2", &cuMemAlloc)) {
    ok &= LoadSymbol(driver, "cuMemAlloc", &cuMemAlloc, error);
  }
  if (!LoadOptional(driver, "cuMemFree_v2", &cuMemFree)) {
    ok &= LoadSymbol(driver, "cuMemFree", &cuMemFree, error);
  }
  ok &= LoadSymbol(driver, "cuMemcpyHtoD", &cuMemcpyHtoD, error);
  ok &= LoadSymbol(driver, "cuMemcpyDtoH", &cuMemcpyDtoH, error);
  LoadOptional(driver, "cuModuleLoadDataEx", &cuModuleLoadDataEx);
  ok &= LoadSymbol(driver, "cuModuleLoadData", &cuModuleLoadData, error);
  LoadOptional(driver, "cuModuleUnload", &cuModuleUnload);
  ok &= LoadSymbol(driver, "cuModuleGetFunction", &cuModuleGetFunction, error);
  ok &= LoadSymbol(driver, "cuLaunchKernel", &cuLaunchKernel, error);
  LoadOptional(driver, "cuGetErrorName", &cuGetErrorName);
  LoadOptional(driver, "cuGetErrorString", &cuGetErrorString);

  if (!ok) {
    Unload();
    return false;
  }

  if (LoadNvrtcLib(&nvrtc)) {
    LoadSymbol(nvrtc, "nvrtcCreateProgram", &nvrtcCreateProgram, nullptr);
    LoadSymbol(nvrtc, "nvrtcCompileProgram", &nvrtcCompileProgram, nullptr);
    LoadSymbol(nvrtc, "nvrtcDestroyProgram", &nvrtcDestroyProgram, nullptr);
    LoadSymbol(nvrtc, "nvrtcGetPTXSize", &nvrtcGetPTXSize, nullptr);
    LoadSymbol(nvrtc, "nvrtcGetPTX", &nvrtcGetPTX, nullptr);
    LoadSymbol(nvrtc, "nvrtcGetProgramLogSize", &nvrtcGetProgramLogSize, nullptr);
    LoadSymbol(nvrtc, "nvrtcGetProgramLog", &nvrtcGetProgramLog, nullptr);
    LoadSymbol(nvrtc, "nvrtcGetErrorString", &nvrtcGetErrorString, nullptr);
    LoadSymbol(nvrtc, "nvrtcVersion", &nvrtcVersion, nullptr);
  }

  return true;
}

void CudaLoader::Unload() {
  if (driver) {
    DlClose(driver);
    driver = nullptr;
  }
  if (nvrtc) {
    DlClose(nvrtc);
    nvrtc = nullptr;
  }
}

std::string CudaErrorString(CUresult err, const CudaLoader* loader) {
  const char* name = nullptr;
  if (loader && loader->cuGetErrorName) {
    loader->cuGetErrorName(err, &name);
  }
  if (name) return name;
  return "CUDA error " + std::to_string(err);
}

std::string NvrtcErrorString(NvrtcResult err, const CudaLoader* loader) {
  if (loader && loader->nvrtcGetErrorString) {
    const char* msg = loader->nvrtcGetErrorString(err);
    if (msg) return msg;
  }
  return "NVRTC error " + std::to_string(err);
}

}  // namespace lattice::runtime::gpu
