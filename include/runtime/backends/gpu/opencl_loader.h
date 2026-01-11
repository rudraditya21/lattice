#ifndef LATTICE_RUNTIME_BACKENDS_GPU_OPENCL_LOADER_H_
#define LATTICE_RUNTIME_BACKENDS_GPU_OPENCL_LOADER_H_

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <CL/cl.h>

#include <string>

#include "runtime/backends/gpu/dynloader.h"

namespace lattice::runtime::gpu {

using OclBuildProgram = cl_int(CL_API_CALL*)(cl_program, cl_uint, const cl_device_id*, const char*,
                                             void(CL_CALLBACK*)(cl_program, void*), void*);
using OclCreateBuffer = cl_mem(CL_API_CALL*)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
using OclCreateCommandQueue = cl_command_queue(CL_API_CALL*)(cl_context, cl_device_id,
                                                             cl_command_queue_properties, cl_int*);
using OclCreateContext = cl_context(CL_API_CALL*)(
    const cl_context_properties*, cl_uint, const cl_device_id*,
    void(CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int*);
using OclCreateKernel = cl_kernel(CL_API_CALL*)(cl_program, const char*, cl_int*);
using OclCreateProgramWithBinary = cl_program(CL_API_CALL*)(cl_context, cl_uint,
                                                            const cl_device_id*, const size_t*,
                                                            const unsigned char**, cl_int*,
                                                            cl_int*);
using OclCreateProgramWithSource = cl_program(CL_API_CALL*)(cl_context, cl_uint, const char**,
                                                            const size_t*, cl_int*);
using OclEnqueueCopyBuffer = cl_int(CL_API_CALL*)(cl_command_queue, cl_mem, cl_mem, size_t, size_t,
                                                  size_t, cl_uint, const cl_event*, cl_event*);
using OclEnqueueFillBuffer = cl_int(CL_API_CALL*)(cl_command_queue, cl_mem, const void*, size_t,
                                                  size_t, size_t, cl_uint, const cl_event*,
                                                  cl_event*);
using OclEnqueueMapBuffer = void*(CL_API_CALL*)(cl_command_queue, cl_mem, cl_bool, cl_map_flags,
                                                size_t, size_t, cl_uint, const cl_event*, cl_event*,
                                                cl_int*);
using OclEnqueueNDRangeKernel = cl_int(CL_API_CALL*)(cl_command_queue, cl_kernel, cl_uint,
                                                     const size_t*, const size_t*, const size_t*,
                                                     cl_uint, const cl_event*, cl_event*);
using OclEnqueueReadBuffer = cl_int(CL_API_CALL*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                                  void*, cl_uint, const cl_event*, cl_event*);
using OclEnqueueUnmapMemObject = cl_int(CL_API_CALL*)(cl_command_queue, cl_mem, void*, cl_uint,
                                                      const cl_event*, cl_event*);
using OclEnqueueWriteBuffer = cl_int(CL_API_CALL*)(cl_command_queue, cl_mem, cl_bool, size_t,
                                                   size_t, const void*, cl_uint, const cl_event*,
                                                   cl_event*);
using OclFinish = cl_int(CL_API_CALL*)(cl_command_queue);
using OclFlush = cl_int(CL_API_CALL*)(cl_command_queue);
using OclGetDeviceIDs = cl_int(CL_API_CALL*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*,
                                             cl_uint*);
using OclGetDeviceInfo = cl_int(CL_API_CALL*)(cl_device_id, cl_device_info, size_t, void*, size_t*);
using OclGetEventInfo = cl_int(CL_API_CALL*)(cl_event, cl_event_info, size_t, void*, size_t*);
using OclGetEventProfilingInfo = cl_int(CL_API_CALL*)(cl_event, cl_profiling_info, size_t, void*,
                                                      size_t*);
using OclGetKernelWorkGroupInfo = cl_int(CL_API_CALL*)(cl_kernel, cl_device_id,
                                                       cl_kernel_work_group_info, size_t, void*,
                                                       size_t*);
using OclGetPlatformIDs = cl_int(CL_API_CALL*)(cl_uint, cl_platform_id*, cl_uint*);
using OclGetPlatformInfo = cl_int(CL_API_CALL*)(cl_platform_id, cl_platform_info, size_t, void*,
                                                size_t*);
using OclGetProgramBuildInfo = cl_int(CL_API_CALL*)(cl_program, cl_device_id, cl_program_build_info,
                                                    size_t, void*, size_t*);
using OclGetProgramInfo = cl_int(CL_API_CALL*)(cl_program, cl_program_info, size_t, void*, size_t*);
using OclLinkProgram = cl_program(CL_API_CALL*)(cl_context, cl_uint, const cl_device_id*,
                                                const char*, cl_uint, const cl_program*,
                                                void(CL_CALLBACK*)(cl_program, void*), void*,
                                                cl_int*);
using OclReleaseCommandQueue = cl_int(CL_API_CALL*)(cl_command_queue);
using OclReleaseContext = cl_int(CL_API_CALL*)(cl_context);
using OclReleaseEvent = cl_int(CL_API_CALL*)(cl_event);
using OclReleaseKernel = cl_int(CL_API_CALL*)(cl_kernel);
using OclReleaseMemObject = cl_int(CL_API_CALL*)(cl_mem);
using OclReleaseProgram = cl_int(CL_API_CALL*)(cl_program);
using OclSetKernelArg = cl_int(CL_API_CALL*)(cl_kernel, cl_uint, size_t, const void*);
using OclWaitForEvents = cl_int(CL_API_CALL*)(cl_uint, const cl_event*);

struct OpenCLLoader {
  DynLib lib = nullptr;

  OclBuildProgram clBuildProgram = nullptr;
  OclCreateBuffer clCreateBuffer = nullptr;
  OclCreateCommandQueue clCreateCommandQueue = nullptr;
  OclCreateContext clCreateContext = nullptr;
  OclCreateKernel clCreateKernel = nullptr;
  OclCreateProgramWithBinary clCreateProgramWithBinary = nullptr;
  OclCreateProgramWithSource clCreateProgramWithSource = nullptr;
  OclEnqueueCopyBuffer clEnqueueCopyBuffer = nullptr;
  OclEnqueueFillBuffer clEnqueueFillBuffer = nullptr;
  OclEnqueueMapBuffer clEnqueueMapBuffer = nullptr;
  OclEnqueueNDRangeKernel clEnqueueNDRangeKernel = nullptr;
  OclEnqueueReadBuffer clEnqueueReadBuffer = nullptr;
  OclEnqueueUnmapMemObject clEnqueueUnmapMemObject = nullptr;
  OclEnqueueWriteBuffer clEnqueueWriteBuffer = nullptr;
  OclFinish clFinish = nullptr;
  OclFlush clFlush = nullptr;
  OclGetDeviceIDs clGetDeviceIDs = nullptr;
  OclGetDeviceInfo clGetDeviceInfo = nullptr;
  OclGetEventInfo clGetEventInfo = nullptr;
  OclGetEventProfilingInfo clGetEventProfilingInfo = nullptr;
  OclGetKernelWorkGroupInfo clGetKernelWorkGroupInfo = nullptr;
  OclGetPlatformIDs clGetPlatformIDs = nullptr;
  OclGetPlatformInfo clGetPlatformInfo = nullptr;
  OclGetProgramBuildInfo clGetProgramBuildInfo = nullptr;
  OclGetProgramInfo clGetProgramInfo = nullptr;
  OclLinkProgram clLinkProgram = nullptr;
  OclReleaseCommandQueue clReleaseCommandQueue = nullptr;
  OclReleaseContext clReleaseContext = nullptr;
  OclReleaseEvent clReleaseEvent = nullptr;
  OclReleaseKernel clReleaseKernel = nullptr;
  OclReleaseMemObject clReleaseMemObject = nullptr;
  OclReleaseProgram clReleaseProgram = nullptr;
  OclSetKernelArg clSetKernelArg = nullptr;
  OclWaitForEvents clWaitForEvents = nullptr;

  bool Load(std::string* error);
  void Unload();
  bool Loaded() const { return lib != nullptr; }
};

std::string OpenCLErrorString(cl_int err);

}  // namespace lattice::runtime::gpu

#endif  // LATTICE_RUNTIME_BACKENDS_GPU_OPENCL_LOADER_H_
