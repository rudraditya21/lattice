#include "runtime/backends/gpu/opencl_loader.h"

#include <cstring>

namespace lattice::runtime::gpu {

namespace {

template <typename T>
bool LoadSymbol(DynLib lib, const char* name, T* out, std::string* error) {
  auto sym = DlSym(lib, name);
  if (!sym) {
    if (error) {
      *error = std::string("Missing OpenCL symbol: ") + name;
    }
    return false;
  }
  *out = reinterpret_cast<T>(sym);
  return true;
}

}  // namespace

bool OpenCLLoader::Load(std::string* error) {
  if (lib) return true;

#if defined(_WIN32)
  const char* libs[] = {"OpenCL.dll", "OpenCL"};
#elif defined(__APPLE__)
  const char* libs[] = {"/System/Library/Frameworks/OpenCL.framework/OpenCL"};
#else
  const char* libs[] = {"libOpenCL.so", "libOpenCL.so.1"};
#endif

  for (const char* name : libs) {
    lib = DlOpen(name);
    if (lib) break;
  }

  if (!lib) {
    if (error) {
      *error = "Failed to load OpenCL runtime: " + DlError();
    }
    return false;
  }

  bool ok = true;
  ok &= LoadSymbol(lib, "clBuildProgram", &clBuildProgram, error);
  ok &= LoadSymbol(lib, "clCreateBuffer", &clCreateBuffer, error);
  ok &= LoadSymbol(lib, "clCreateCommandQueue", &clCreateCommandQueue, error);
  ok &= LoadSymbol(lib, "clCreateContext", &clCreateContext, error);
  ok &= LoadSymbol(lib, "clCreateKernel", &clCreateKernel, error);
  ok &= LoadSymbol(lib, "clCreateProgramWithBinary", &clCreateProgramWithBinary, error);
  ok &= LoadSymbol(lib, "clCreateProgramWithSource", &clCreateProgramWithSource, error);
  ok &= LoadSymbol(lib, "clEnqueueCopyBuffer", &clEnqueueCopyBuffer, error);
  LoadSymbol(lib, "clEnqueueFillBuffer", &clEnqueueFillBuffer, nullptr);
  LoadSymbol(lib, "clEnqueueMapBuffer", &clEnqueueMapBuffer, nullptr);
  ok &= LoadSymbol(lib, "clEnqueueNDRangeKernel", &clEnqueueNDRangeKernel, error);
  ok &= LoadSymbol(lib, "clEnqueueReadBuffer", &clEnqueueReadBuffer, error);
  LoadSymbol(lib, "clEnqueueUnmapMemObject", &clEnqueueUnmapMemObject, nullptr);
  ok &= LoadSymbol(lib, "clEnqueueWriteBuffer", &clEnqueueWriteBuffer, error);
  ok &= LoadSymbol(lib, "clFinish", &clFinish, error);
  ok &= LoadSymbol(lib, "clFlush", &clFlush, error);
  ok &= LoadSymbol(lib, "clGetDeviceIDs", &clGetDeviceIDs, error);
  ok &= LoadSymbol(lib, "clGetDeviceInfo", &clGetDeviceInfo, error);
  ok &= LoadSymbol(lib, "clGetEventInfo", &clGetEventInfo, error);
  ok &= LoadSymbol(lib, "clGetEventProfilingInfo", &clGetEventProfilingInfo, error);
  ok &= LoadSymbol(lib, "clGetKernelWorkGroupInfo", &clGetKernelWorkGroupInfo, error);
  ok &= LoadSymbol(lib, "clGetPlatformIDs", &clGetPlatformIDs, error);
  ok &= LoadSymbol(lib, "clGetPlatformInfo", &clGetPlatformInfo, error);
  ok &= LoadSymbol(lib, "clGetProgramBuildInfo", &clGetProgramBuildInfo, error);
  ok &= LoadSymbol(lib, "clGetProgramInfo", &clGetProgramInfo, error);
  ok &= LoadSymbol(lib, "clLinkProgram", &clLinkProgram, error);
  ok &= LoadSymbol(lib, "clReleaseCommandQueue", &clReleaseCommandQueue, error);
  ok &= LoadSymbol(lib, "clReleaseContext", &clReleaseContext, error);
  ok &= LoadSymbol(lib, "clReleaseEvent", &clReleaseEvent, error);
  ok &= LoadSymbol(lib, "clReleaseKernel", &clReleaseKernel, error);
  ok &= LoadSymbol(lib, "clReleaseMemObject", &clReleaseMemObject, error);
  ok &= LoadSymbol(lib, "clReleaseProgram", &clReleaseProgram, error);
  ok &= LoadSymbol(lib, "clSetKernelArg", &clSetKernelArg, error);
  ok &= LoadSymbol(lib, "clWaitForEvents", &clWaitForEvents, error);

  if (!ok) {
    Unload();
    return false;
  }

  return true;
}

void OpenCLLoader::Unload() {
  if (lib) {
    DlClose(lib);
    lib = nullptr;
  }
}

std::string OpenCLErrorString(cl_int err) {
  switch (err) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
    case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_GLOBAL_WORK_SIZE:
      return "CL_INVALID_GLOBAL_WORK_SIZE";
    default:
      break;
  }
  return "CL_UNKNOWN_ERROR";
}

}  // namespace lattice::runtime::gpu
