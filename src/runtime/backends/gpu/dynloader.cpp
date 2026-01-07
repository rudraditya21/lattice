#include "runtime/backends/gpu/dynloader.h"

namespace lattice::runtime::gpu {

#if defined(_WIN32)

DynLib DlOpen(const char* filename) {
  return LoadLibraryA(filename);
}

bool DlClose(DynLib handle) {
  return FreeLibrary(handle) != 0;
}

DynFunc DlSym(DynLib handle, const char* symbol) {
  return GetProcAddress(handle, symbol);
}

std::string DlError() {
  DWORD rc = GetLastError();
  char* msg = nullptr;
  FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, rc, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPSTR>(&msg), 0,
      nullptr);
  std::string out = msg ? msg : "";
  if (msg) LocalFree(msg);
  return out;
}

#else

DynLib DlOpen(const char* filename) {
  return dlopen(filename, RTLD_NOW);
}

bool DlClose(DynLib handle) {
  return dlclose(handle) == 0;
}

DynFunc DlSym(DynLib handle, const char* symbol) {
  return dlsym(handle, symbol);
}

std::string DlError() {
  const char* msg = dlerror();
  return msg ? msg : "";
}

#endif

}  // namespace lattice::runtime::gpu
