#ifndef LATTICE_RUNTIME_BACKENDS_GPU_HIP_LOADER_H_
#define LATTICE_RUNTIME_BACKENDS_GPU_HIP_LOADER_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "runtime/backends/gpu/dynloader.h"

namespace lattice::runtime::gpu {

using hipError_t = int;
using hipDevice_t = int;
using hipCtx_t = void*;
using hipStream_t = void*;
using hipModule_t = void*;
using hipFunction_t = void*;
using hipEvent_t = void*;
using hipDeviceptr_t = void*;

constexpr hipError_t hipSuccess = 0;

enum hipMemcpyKind {
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
};

using HipInit = hipError_t (*)(unsigned int);
using HipDriverGetVersion = hipError_t (*)(int*);
using HipGetDeviceCount = hipError_t (*)(int*);
using HipDeviceGet = hipError_t (*)(hipDevice_t*, int);
using HipDeviceGetName = hipError_t (*)(char*, int, hipDevice_t);
using HipDeviceTotalMem = hipError_t (*)(size_t*, hipDevice_t);
using HipDeviceGetAttribute = hipError_t (*)(int*, int, hipDevice_t);
using HipCtxCreate = hipError_t (*)(hipCtx_t*, unsigned int, hipDevice_t);
using HipCtxDestroy = hipError_t (*)(hipCtx_t);
using HipCtxSetCurrent = hipError_t (*)(hipCtx_t);
using HipStreamCreate = hipError_t (*)(hipStream_t*);
using HipStreamDestroy = hipError_t (*)(hipStream_t);
using HipStreamSynchronize = hipError_t (*)(hipStream_t);
using HipMalloc = hipError_t (*)(void**, size_t);
using HipFree = hipError_t (*)(void*);
using HipMemcpy = hipError_t (*)(void*, const void*, size_t, hipMemcpyKind);
using HipModuleLoadData = hipError_t (*)(hipModule_t*, const void*);
using HipModuleUnload = hipError_t (*)(hipModule_t);
using HipModuleGetFunction = hipError_t (*)(hipFunction_t*, hipModule_t, const char*);
using HipModuleLaunchKernel = hipError_t (*)(hipFunction_t, unsigned int, unsigned int,
                                             unsigned int, unsigned int, unsigned int, unsigned int,
                                             unsigned int, hipStream_t, void**, void**);
using HipGetErrorString = const char* (*)(hipError_t);
using HipRuntimeGetVersion = hipError_t (*)(int*);

using HiprtcResult = int;
using hiprtcProgram = void*;
using HiprtcCreateProgram = HiprtcResult (*)(hiprtcProgram*, const char*, const char*, int,
                                             const char* const*, const char* const*);
using HiprtcCompileProgram = HiprtcResult (*)(hiprtcProgram, int, const char* const*);
using HiprtcDestroyProgram = HiprtcResult (*)(hiprtcProgram*);
using HiprtcGetCodeSize = HiprtcResult (*)(hiprtcProgram, size_t*);
using HiprtcGetCode = HiprtcResult (*)(hiprtcProgram, char*);
using HiprtcGetProgramLogSize = HiprtcResult (*)(hiprtcProgram, size_t*);
using HiprtcGetProgramLog = HiprtcResult (*)(hiprtcProgram, char*);
using HiprtcGetErrorString = const char* (*)(HiprtcResult);

struct HipLoader {
  DynLib driver = nullptr;
  DynLib hiprtc = nullptr;

  HipInit hipInit = nullptr;
  HipDriverGetVersion hipDriverGetVersion = nullptr;
  HipGetDeviceCount hipGetDeviceCount = nullptr;
  HipDeviceGet hipDeviceGet = nullptr;
  HipDeviceGetName hipDeviceGetName = nullptr;
  HipDeviceTotalMem hipDeviceTotalMem = nullptr;
  HipDeviceGetAttribute hipDeviceGetAttribute = nullptr;
  HipCtxCreate hipCtxCreate = nullptr;
  HipCtxDestroy hipCtxDestroy = nullptr;
  HipCtxSetCurrent hipCtxSetCurrent = nullptr;
  HipStreamCreate hipStreamCreate = nullptr;
  HipStreamDestroy hipStreamDestroy = nullptr;
  HipStreamSynchronize hipStreamSynchronize = nullptr;
  HipMalloc hipMalloc = nullptr;
  HipFree hipFree = nullptr;
  HipMemcpy hipMemcpy = nullptr;
  HipModuleLoadData hipModuleLoadData = nullptr;
  HipModuleUnload hipModuleUnload = nullptr;
  HipModuleGetFunction hipModuleGetFunction = nullptr;
  HipModuleLaunchKernel hipModuleLaunchKernel = nullptr;
  HipGetErrorString hipGetErrorString = nullptr;
  HipRuntimeGetVersion hipRuntimeGetVersion = nullptr;

  HiprtcCreateProgram hiprtcCreateProgram = nullptr;
  HiprtcCompileProgram hiprtcCompileProgram = nullptr;
  HiprtcDestroyProgram hiprtcDestroyProgram = nullptr;
  HiprtcGetCodeSize hiprtcGetCodeSize = nullptr;
  HiprtcGetCode hiprtcGetCode = nullptr;
  HiprtcGetProgramLogSize hiprtcGetProgramLogSize = nullptr;
  HiprtcGetProgramLog hiprtcGetProgramLog = nullptr;
  HiprtcGetErrorString hiprtcGetErrorString = nullptr;

  bool Load(std::string* error);
  void Unload();
  bool Loaded() const { return driver != nullptr; }
  bool HiprtcLoaded() const { return hiprtc != nullptr; }
};

std::string HipErrorString(hipError_t err, const HipLoader* loader);
std::string HiprtcErrorString(HiprtcResult err, const HipLoader* loader);

}  // namespace lattice::runtime::gpu

#endif  // LATTICE_RUNTIME_BACKENDS_GPU_HIP_LOADER_H_
