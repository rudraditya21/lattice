#include "runtime/backends/gpu/hipblas_loader.h"

#include <cstring>

namespace lattice::runtime::gpu {

namespace {

template <typename T>
bool LoadSymbol(DynLib lib, const char* name, T* out, std::string* error) {
    auto sym = DlSym(lib, name);
    if (!sym) {
        if (error) {
            *error = std::string("Missing hipBLAS symbol: ") + name;
        }
        return false;
    }
    *out = reinterpret_cast<T>(sym);
    return true;
}

template <typename T>
bool LoadOptional(DynLib lib, const char* name, T* out) {
    auto sym = DlSym(lib, name);
    if (!sym)
        return false;
    *out = reinterpret_cast<T>(sym);
    return true;
}

bool LoadHipblasLib(DynLib* lib, std::string* error) {
#if defined(_WIN32)
    const char* libs[] = {"hipblas.dll"};
#elif defined(__APPLE__)
    const char* libs[] = {"libhipblas.dylib"};
#else
    const char* libs[] = {"libhipblas.so", "libhipblas.so.0"};
#endif
    for (const char* name : libs) {
        *lib = DlOpen(name);
        if (*lib)
            return true;
    }
    if (error) {
        *error = "Failed to load hipBLAS: " + DlError();
    }
    return false;
}

}  // namespace

bool HipblasLoader::Load(std::string* error) {
    if (lib)
        return true;
    if (!LoadHipblasLib(&lib, error)) {
        return false;
    }
    bool ok = true;
    ok &= LoadSymbol(lib, "hipblasCreate", &hipblasCreate, error);
    ok &= LoadSymbol(lib, "hipblasDestroy", &hipblasDestroy, error);
    ok &= LoadSymbol(lib, "hipblasSetStream", &hipblasSetStream, error);
    LoadOptional(lib, "hipblasScopy", &hipblasScopy);
    LoadOptional(lib, "hipblasDcopy", &hipblasDcopy);
    LoadOptional(lib, "hipblasSaxpy", &hipblasSaxpy);
    LoadOptional(lib, "hipblasDaxpy", &hipblasDaxpy);
    ok &= LoadSymbol(lib, "hipblasSgemm", &hipblasSgemm, error);
    ok &= LoadSymbol(lib, "hipblasDgemm", &hipblasDgemm, error);
    if (!ok) {
        Unload();
        return false;
    }
    return true;
}

void HipblasLoader::Unload() {
    if (lib) {
        DlClose(lib);
        lib = nullptr;
    }
}

std::string HipblasErrorString(hipblasStatus_t status) {
    switch (status) {
        case HIPBLAS_STATUS_SUCCESS:
            return "HIPBLAS_STATUS_SUCCESS";
        case 1:
            return "HIPBLAS_STATUS_NOT_INITIALIZED";
        case 2:
            return "HIPBLAS_STATUS_ALLOC_FAILED";
        case 3:
            return "HIPBLAS_STATUS_INVALID_VALUE";
        case 4:
            return "HIPBLAS_STATUS_MAPPING_ERROR";
        case 5:
            return "HIPBLAS_STATUS_EXECUTION_FAILED";
        case 6:
            return "HIPBLAS_STATUS_INTERNAL_ERROR";
        case 7:
            return "HIPBLAS_STATUS_NOT_SUPPORTED";
        case 8:
            return "HIPBLAS_STATUS_ARCH_MISMATCH";
        default:
            return "HIPBLAS_STATUS_" + std::to_string(status);
    }
}

}  // namespace lattice::runtime::gpu
