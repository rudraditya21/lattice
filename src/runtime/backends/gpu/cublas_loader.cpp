#include "runtime/backends/gpu/cublas_loader.h"

#include <cstring>

namespace lattice::runtime::gpu {

namespace {

template <typename T>
bool LoadSymbol(DynLib lib, const char* name, T* out, std::string* error) {
    auto sym = DlSym(lib, name);
    if (!sym) {
        if (error) {
            *error = std::string("Missing cuBLAS symbol: ") + name;
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

bool LoadCublasLib(DynLib* lib, std::string* error) {
#if defined(_WIN32)
    const char* libs[] = {"cublas64_12.dll", "cublas64_11.dll",
                          "cublas64_10.dll", "cublas64.dll"};
#elif defined(__APPLE__)
    const char* libs[] = {"libcublas.dylib",
                          "/usr/local/cuda/lib/libcublas.dylib"};
#else
    const char* libs[] = {"libcublas.so", "libcublas.so.12", "libcublas.so.11",
                          "libcublas.so.10"};
#endif
    for (const char* name : libs) {
        *lib = DlOpen(name);
        if (*lib)
            return true;
    }
    if (error) {
        *error = "Failed to load cuBLAS: " + DlError();
    }
    return false;
}

}  // namespace

bool CublasLoader::Load(std::string* error) {
    if (lib)
        return true;
    if (!LoadCublasLib(&lib, error)) {
        return false;
    }
    bool ok = true;
    if (!LoadOptional(lib, "cublasCreate_v2", &cublasCreate)) {
        ok &= LoadSymbol(lib, "cublasCreate", &cublasCreate, error);
    }
    if (!LoadOptional(lib, "cublasDestroy_v2", &cublasDestroy)) {
        ok &= LoadSymbol(lib, "cublasDestroy", &cublasDestroy, error);
    }
    if (!LoadOptional(lib, "cublasSetStream_v2", &cublasSetStream)) {
        ok &= LoadSymbol(lib, "cublasSetStream", &cublasSetStream, error);
    }
    if (!LoadOptional(lib, "cublasScopy_v2", &cublasScopy)) {
        LoadOptional(lib, "cublasScopy", &cublasScopy);
    }
    if (!LoadOptional(lib, "cublasDcopy_v2", &cublasDcopy)) {
        LoadOptional(lib, "cublasDcopy", &cublasDcopy);
    }
    if (!LoadOptional(lib, "cublasSaxpy_v2", &cublasSaxpy)) {
        LoadOptional(lib, "cublasSaxpy", &cublasSaxpy);
    }
    if (!LoadOptional(lib, "cublasDaxpy_v2", &cublasDaxpy)) {
        LoadOptional(lib, "cublasDaxpy", &cublasDaxpy);
    }
    if (!LoadOptional(lib, "cublasSgemm_v2", &cublasSgemm)) {
        ok &= LoadSymbol(lib, "cublasSgemm", &cublasSgemm, error);
    }
    if (!LoadOptional(lib, "cublasDgemm_v2", &cublasDgemm)) {
        ok &= LoadSymbol(lib, "cublasDgemm", &cublasDgemm, error);
    }
    if (!ok) {
        Unload();
        return false;
    }
    return true;
}

void CublasLoader::Unload() {
    if (lib) {
        DlClose(lib);
        lib = nullptr;
    }
}

std::string CublasErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case 1:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case 3:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case 4:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case 5:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case 6:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case 7:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case 8:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case 9:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case 10:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "CUBLAS_STATUS_" + std::to_string(status);
    }
}

}  // namespace lattice::runtime::gpu
