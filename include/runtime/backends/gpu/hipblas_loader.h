#ifndef LATTICE_RUNTIME_BACKENDS_GPU_HIPBLAS_LOADER_H_
#define LATTICE_RUNTIME_BACKENDS_GPU_HIPBLAS_LOADER_H_

#include <string>

#include "runtime/backends/gpu/dynloader.h"
#include "runtime/backends/gpu/hip_loader.h"

namespace lattice::runtime::gpu {

using hipblasHandle_t = void*;
using hipblasStatus_t = int;
using hipblasOperation_t = int;

constexpr hipblasStatus_t HIPBLAS_STATUS_SUCCESS = 0;
constexpr hipblasOperation_t HIPBLAS_OP_N = 0;
constexpr hipblasOperation_t HIPBLAS_OP_T = 1;
constexpr hipblasOperation_t HIPBLAS_OP_C = 2;

using HipblasCreate = hipblasStatus_t (*)(hipblasHandle_t*);
using HipblasDestroy = hipblasStatus_t (*)(hipblasHandle_t);
using HipblasSetStream = hipblasStatus_t (*)(hipblasHandle_t, hipStream_t);
using HipblasScopy =
    hipblasStatus_t (*)(hipblasHandle_t, int, const float*, int, float*, int);
using HipblasDcopy =
    hipblasStatus_t (*)(hipblasHandle_t, int, const double*, int, double*, int);
using HipblasSaxpy = hipblasStatus_t (*)(hipblasHandle_t,
                                         int,
                                         const float*,
                                         const float*,
                                         int,
                                         float*,
                                         int);
using HipblasDaxpy = hipblasStatus_t (*)(hipblasHandle_t,
                                         int,
                                         const double*,
                                         const double*,
                                         int,
                                         double*,
                                         int);
using HipblasSgemm = hipblasStatus_t (*)(hipblasHandle_t,
                                         hipblasOperation_t,
                                         hipblasOperation_t,
                                         int,
                                         int,
                                         int,
                                         const float*,
                                         const float*,
                                         int,
                                         const float*,
                                         int,
                                         const float*,
                                         float*,
                                         int);
using HipblasDgemm = hipblasStatus_t (*)(hipblasHandle_t,
                                         hipblasOperation_t,
                                         hipblasOperation_t,
                                         int,
                                         int,
                                         int,
                                         const double*,
                                         const double*,
                                         int,
                                         const double*,
                                         int,
                                         const double*,
                                         double*,
                                         int);

struct HipblasLoader {
    DynLib lib = nullptr;

    HipblasCreate hipblasCreate = nullptr;
    HipblasDestroy hipblasDestroy = nullptr;
    HipblasSetStream hipblasSetStream = nullptr;
    HipblasScopy hipblasScopy = nullptr;
    HipblasDcopy hipblasDcopy = nullptr;
    HipblasSaxpy hipblasSaxpy = nullptr;
    HipblasDaxpy hipblasDaxpy = nullptr;
    HipblasSgemm hipblasSgemm = nullptr;
    HipblasDgemm hipblasDgemm = nullptr;

    bool Load(std::string* error);
    void Unload();
    bool Loaded() const { return lib != nullptr; }
};

std::string HipblasErrorString(hipblasStatus_t status);

}  // namespace lattice::runtime::gpu

#endif  // LATTICE_RUNTIME_BACKENDS_GPU_HIPBLAS_LOADER_H_
