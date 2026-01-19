#ifndef LATTICE_RUNTIME_BACKENDS_GPU_CLBLAST_LOADER_H_
#define LATTICE_RUNTIME_BACKENDS_GPU_CLBLAST_LOADER_H_

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#endif

#include <CL/cl.h>

#include <string>

#include "runtime/backends/gpu/dynloader.h"

namespace lattice::runtime::gpu {

using CLBlastStatusCode = int;
using CLBlastLayout = int;
using CLBlastTranspose = int;

constexpr CLBlastStatusCode CLBLAST_STATUS_SUCCESS = 0;

constexpr CLBlastLayout CLBLAST_LAYOUT_ROW_MAJOR = 101;
constexpr CLBlastLayout CLBLAST_LAYOUT_COL_MAJOR = 102;

constexpr CLBlastTranspose CLBLAST_TRANSPOSE_NO = 111;
constexpr CLBlastTranspose CLBLAST_TRANSPOSE_YES = 112;
constexpr CLBlastTranspose CLBLAST_TRANSPOSE_CONJ = 113;

using ClblastSgemm = CLBlastStatusCode (*)(CLBlastLayout,
                                           CLBlastTranspose,
                                           CLBlastTranspose,
                                           size_t,
                                           size_t,
                                           size_t,
                                           float,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           float,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           cl_command_queue*,
                                           cl_event*);
using ClblastDgemm = CLBlastStatusCode (*)(CLBlastLayout,
                                           CLBlastTranspose,
                                           CLBlastTranspose,
                                           size_t,
                                           size_t,
                                           size_t,
                                           double,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           double,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           cl_command_queue*,
                                           cl_event*);
using ClblastScopy = CLBlastStatusCode (*)(size_t,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           cl_command_queue*,
                                           cl_event*);
using ClblastDcopy = CLBlastStatusCode (*)(size_t,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           cl_command_queue*,
                                           cl_event*);
using ClblastSaxpy = CLBlastStatusCode (*)(size_t,
                                           float,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           cl_command_queue*,
                                           cl_event*);
using ClblastDaxpy = CLBlastStatusCode (*)(size_t,
                                           double,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           cl_mem,
                                           size_t,
                                           size_t,
                                           cl_command_queue*,
                                           cl_event*);

struct ClblastLoader {
    DynLib lib = nullptr;

    ClblastSgemm clblastSgemm = nullptr;
    ClblastDgemm clblastDgemm = nullptr;
    ClblastScopy clblastScopy = nullptr;
    ClblastDcopy clblastDcopy = nullptr;
    ClblastSaxpy clblastSaxpy = nullptr;
    ClblastDaxpy clblastDaxpy = nullptr;

    bool Load(std::string* error);
    void Unload();
    bool Loaded() const { return lib != nullptr; }
};

std::string ClblastErrorString(CLBlastStatusCode status);

}  // namespace lattice::runtime::gpu

#endif  // LATTICE_RUNTIME_BACKENDS_GPU_CLBLAST_LOADER_H_
