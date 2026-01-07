#ifndef LATTICE_RUNTIME_BACKENDS_GPU_DYNLOADER_H_
#define LATTICE_RUNTIME_BACKENDS_GPU_DYNLOADER_H_

#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace lattice::runtime::gpu {

#if defined(_WIN32)
using DynLib = HMODULE;
using DynFunc = FARPROC;
#else
using DynLib = void*;
using DynFunc = void*;
#endif

DynLib DlOpen(const char* filename);
bool DlClose(DynLib handle);
DynFunc DlSym(DynLib handle, const char* symbol);
std::string DlError();

}  // namespace lattice::runtime::gpu

#endif  // LATTICE_RUNTIME_BACKENDS_GPU_DYNLOADER_H_
