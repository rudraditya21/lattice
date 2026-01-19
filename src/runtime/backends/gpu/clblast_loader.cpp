#include "runtime/backends/gpu/clblast_loader.h"

#include <cstring>

namespace lattice::runtime::gpu {

namespace {

template <typename T>
bool LoadSymbol(DynLib lib, const char* name, T* out, std::string* error) {
    auto sym = DlSym(lib, name);
    if (!sym) {
        if (error) {
            *error = std::string("Missing CLBlast symbol: ") + name;
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

bool LoadClblastLib(DynLib* lib, std::string* error) {
#if defined(_WIN32)
    const char* libs[] = {"clblast.dll", "CLBlast.dll"};
#elif defined(__APPLE__)
    const char* libs[] = {"libclblast.dylib"};
#else
    const char* libs[] = {"libclblast.so", "libclblast.so.1"};
#endif
    for (const char* name : libs) {
        *lib = DlOpen(name);
        if (*lib)
            return true;
    }
    if (error) {
        *error = "Failed to load CLBlast: " + DlError();
    }
    return false;
}

}  // namespace

bool ClblastLoader::Load(std::string* error) {
    if (lib)
        return true;
    if (!LoadClblastLib(&lib, error)) {
        return false;
    }
    bool ok = true;
    ok &= LoadSymbol(lib, "CLBlastSgemm", &clblastSgemm, error);
    LoadOptional(lib, "CLBlastDgemm", &clblastDgemm);
    LoadOptional(lib, "CLBlastScopy", &clblastScopy);
    LoadOptional(lib, "CLBlastDcopy", &clblastDcopy);
    LoadOptional(lib, "CLBlastSaxpy", &clblastSaxpy);
    LoadOptional(lib, "CLBlastDaxpy", &clblastDaxpy);
    if (!ok) {
        Unload();
        return false;
    }
    return true;
}

void ClblastLoader::Unload() {
    if (lib) {
        DlClose(lib);
        lib = nullptr;
    }
}

std::string ClblastErrorString(CLBlastStatusCode status) {
    switch (status) {
        case CLBLAST_STATUS_SUCCESS:
            return "CLBlastSuccess";
        case -1:
            return "CLBlastOpenCLCompilerNotAvailable";
        case -2:
            return "CLBlastTempBufferAllocFailure";
        case -3:
            return "CLBlastOpenCLOutOfResources";
        case -4:
            return "CLBlastOpenCLBuildProgramFailure";
        case -5:
            return "CLBlastInvalidValue";
        case -6:
            return "CLBlastInvalidCommandQueue";
        case -7:
            return "CLBlastInvalidMemObject";
        case -8:
            return "CLBlastInvalidBinary";
        case -9:
            return "CLBlastInvalidBuildOptions";
        case -10:
            return "CLBlastInvalidProgram";
        case -11:
            return "CLBlastInvalidProgramExecutable";
        case -12:
            return "CLBlastInvalidKernelName";
        case -13:
            return "CLBlastInvalidKernelDefinition";
        case -14:
            return "CLBlastInvalidKernel";
        case -15:
            return "CLBlastInvalidArgIndex";
        case -16:
            return "CLBlastInvalidArgValue";
        case -17:
            return "CLBlastInvalidArgSize";
        case -18:
            return "CLBlastInvalidKernelArgs";
        case -19:
            return "CLBlastInvalidLocalNumDimensions";
        case -20:
            return "CLBlastInvalidLocalThreadsTotal";
        case -21:
            return "CLBlastInvalidLocalThreadsDim";
        case -22:
            return "CLBlastInvalidGlobalOffset";
        case -23:
            return "CLBlastInvalidEventWaitList";
        case -24:
            return "CLBlastInvalidEvent";
        case -25:
            return "CLBlastInvalidOperation";
        case -26:
            return "CLBlastInvalidMatrixA";
        case -27:
            return "CLBlastInvalidMatrixB";
        case -28:
            return "CLBlastInvalidMatrixC";
        case -29:
            return "CLBlastInvalidVectorX";
        case -30:
            return "CLBlastInvalidVectorY";
        case -31:
            return "CLBlastInvalidDimension";
        case -32:
            return "CLBlastInvalidLeadDimA";
        case -33:
            return "CLBlastInvalidLeadDimB";
        case -34:
            return "CLBlastInvalidLeadDimC";
        case -35:
            return "CLBlastInvalidIncrementX";
        case -36:
            return "CLBlastInvalidIncrementY";
        case -37:
            return "CLBlastInsufficientMemory";
        case -38:
            return "CLBlastInvalidBatchCount";
        default:
            return "CLBlastStatus_" + std::to_string(status);
    }
}

}  // namespace lattice::runtime::gpu
