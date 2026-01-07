#ifndef LATTICE_RUNTIME_BACKENDS_GPU_CUDA_LOADER_H_
#define LATTICE_RUNTIME_BACKENDS_GPU_CUDA_LOADER_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "runtime/backends/gpu/dynloader.h"

namespace lattice::runtime::gpu {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUstream = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUevent = void*;
using CUdeviceptr = uint64_t;
using CUjit_option = int;

constexpr CUresult CUDA_SUCCESS = 0;

using CuInit = CUresult (*)(unsigned int);
using CuDriverGetVersion = CUresult (*)(int*);
using CuDeviceGetCount = CUresult (*)(int*);
using CuDeviceGet = CUresult (*)(CUdevice*, int);
using CuDeviceGetName = CUresult (*)(char*, int, CUdevice);
using CuDeviceComputeCapability = CUresult (*)(int*, int*, CUdevice);
using CuDeviceTotalMem = CUresult (*)(size_t*, CUdevice);
using CuDeviceGetAttribute = CUresult (*)(int*, int, CUdevice);
using CuCtxCreate = CUresult (*)(CUcontext*, unsigned int, CUdevice);
using CuCtxDestroy = CUresult (*)(CUcontext);
using CuCtxSetCurrent = CUresult (*)(CUcontext);
using CuStreamCreate = CUresult (*)(CUstream*, unsigned int);
using CuStreamDestroy = CUresult (*)(CUstream);
using CuStreamSynchronize = CUresult (*)(CUstream);
using CuMemAlloc = CUresult (*)(CUdeviceptr*, size_t);
using CuMemFree = CUresult (*)(CUdeviceptr);
using CuMemcpyHtoD = CUresult (*)(CUdeviceptr, const void*, size_t);
using CuMemcpyDtoH = CUresult (*)(void*, CUdeviceptr, size_t);
using CuModuleLoadData = CUresult (*)(CUmodule*, const void*);
using CuModuleLoadDataEx = CUresult (*)(CUmodule*, const void*, unsigned int, CUjit_option*,
                                        void**);
using CuModuleUnload = CUresult (*)(CUmodule);
using CuModuleGetFunction = CUresult (*)(CUfunction*, CUmodule, const char*);
using CuLaunchKernel = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int, unsigned int,
                                    CUstream, void**, void**);
using CuGetErrorName = CUresult (*)(CUresult, const char**);
using CuGetErrorString = CUresult (*)(CUresult, const char**);

using NvrtcResult = int;
using nvrtcProgram = void*;
using NvrtcCreateProgram = NvrtcResult (*)(nvrtcProgram*, const char*, const char*, int,
                                           const char* const*, const char* const*);
using NvrtcCompileProgram = NvrtcResult (*)(nvrtcProgram, int, const char* const*);
using NvrtcDestroyProgram = NvrtcResult (*)(nvrtcProgram*);
using NvrtcGetPTXSize = NvrtcResult (*)(nvrtcProgram, size_t*);
using NvrtcGetPTX = NvrtcResult (*)(nvrtcProgram, char*);
using NvrtcGetProgramLogSize = NvrtcResult (*)(nvrtcProgram, size_t*);
using NvrtcGetProgramLog = NvrtcResult (*)(nvrtcProgram, char*);
using NvrtcGetErrorString = const char* (*)(NvrtcResult);

struct CudaLoader {
  DynLib driver = nullptr;
  DynLib nvrtc = nullptr;

  CuInit cuInit = nullptr;
  CuDriverGetVersion cuDriverGetVersion = nullptr;
  CuDeviceGetCount cuDeviceGetCount = nullptr;
  CuDeviceGet cuDeviceGet = nullptr;
  CuDeviceGetName cuDeviceGetName = nullptr;
  CuDeviceComputeCapability cuDeviceComputeCapability = nullptr;
  CuDeviceTotalMem cuDeviceTotalMem = nullptr;
  CuDeviceGetAttribute cuDeviceGetAttribute = nullptr;
  CuCtxCreate cuCtxCreate = nullptr;
  CuCtxDestroy cuCtxDestroy = nullptr;
  CuCtxSetCurrent cuCtxSetCurrent = nullptr;
  CuStreamCreate cuStreamCreate = nullptr;
  CuStreamDestroy cuStreamDestroy = nullptr;
  CuStreamSynchronize cuStreamSynchronize = nullptr;
  CuMemAlloc cuMemAlloc = nullptr;
  CuMemFree cuMemFree = nullptr;
  CuMemcpyHtoD cuMemcpyHtoD = nullptr;
  CuMemcpyDtoH cuMemcpyDtoH = nullptr;
  CuModuleLoadData cuModuleLoadData = nullptr;
  CuModuleLoadDataEx cuModuleLoadDataEx = nullptr;
  CuModuleUnload cuModuleUnload = nullptr;
  CuModuleGetFunction cuModuleGetFunction = nullptr;
  CuLaunchKernel cuLaunchKernel = nullptr;
  CuGetErrorName cuGetErrorName = nullptr;
  CuGetErrorString cuGetErrorString = nullptr;

  NvrtcCreateProgram nvrtcCreateProgram = nullptr;
  NvrtcCompileProgram nvrtcCompileProgram = nullptr;
  NvrtcDestroyProgram nvrtcDestroyProgram = nullptr;
  NvrtcGetPTXSize nvrtcGetPTXSize = nullptr;
  NvrtcGetPTX nvrtcGetPTX = nullptr;
  NvrtcGetProgramLogSize nvrtcGetProgramLogSize = nullptr;
  NvrtcGetProgramLog nvrtcGetProgramLog = nullptr;
  NvrtcGetErrorString nvrtcGetErrorString = nullptr;

  bool Load(std::string* error);
  void Unload();
  bool Loaded() const { return driver != nullptr; }
  bool NvrtcLoaded() const { return nvrtc != nullptr; }
};

std::string CudaErrorString(CUresult err, const CudaLoader* loader);
std::string NvrtcErrorString(NvrtcResult err, const CudaLoader* loader);

}  // namespace lattice::runtime::gpu

#endif  // LATTICE_RUNTIME_BACKENDS_GPU_CUDA_LOADER_H_
