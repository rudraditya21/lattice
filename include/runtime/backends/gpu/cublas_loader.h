#ifndef LATTICE_RUNTIME_BACKENDS_GPU_CUBLAS_LOADER_H_
#define LATTICE_RUNTIME_BACKENDS_GPU_CUBLAS_LOADER_H_

#include <string>

#include "runtime/backends/gpu/cuda_loader.h"
#include "runtime/backends/gpu/dynloader.h"

namespace lattice::runtime::gpu {

using cublasHandle_t = void*;
using cublasStatus_t = int;
using cublasOperation_t = int;

constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
constexpr cublasOperation_t CUBLAS_OP_N = 0;
constexpr cublasOperation_t CUBLAS_OP_T = 1;
constexpr cublasOperation_t CUBLAS_OP_C = 2;

using CublasCreate = cublasStatus_t (*)(cublasHandle_t*);
using CublasDestroy = cublasStatus_t (*)(cublasHandle_t);
using CublasSetStream = cublasStatus_t (*)(cublasHandle_t, CUstream);
using CublasScopy =
    cublasStatus_t (*)(cublasHandle_t, int, const float*, int, float*, int);
using CublasDcopy =
    cublasStatus_t (*)(cublasHandle_t, int, const double*, int, double*, int);
using CublasSaxpy = cublasStatus_t (*)(cublasHandle_t,
                                       int,
                                       const float*,
                                       const float*,
                                       int,
                                       float*,
                                       int);
using CublasDaxpy = cublasStatus_t (*)(cublasHandle_t,
                                       int,
                                       const double*,
                                       const double*,
                                       int,
                                       double*,
                                       int);
using CublasSgemm = cublasStatus_t (*)(cublasHandle_t,
                                       cublasOperation_t,
                                       cublasOperation_t,
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
using CublasDgemm = cublasStatus_t (*)(cublasHandle_t,
                                       cublasOperation_t,
                                       cublasOperation_t,
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

struct CublasLoader {
    DynLib lib = nullptr;

    CublasCreate cublasCreate = nullptr;
    CublasDestroy cublasDestroy = nullptr;
    CublasSetStream cublasSetStream = nullptr;
    CublasScopy cublasScopy = nullptr;
    CublasDcopy cublasDcopy = nullptr;
    CublasSaxpy cublasSaxpy = nullptr;
    CublasDaxpy cublasDaxpy = nullptr;
    CublasSgemm cublasSgemm = nullptr;
    CublasDgemm cublasDgemm = nullptr;

    bool Load(std::string* error);
    void Unload();
    bool Loaded() const { return lib != nullptr; }
};

std::string CublasErrorString(cublasStatus_t status);

}  // namespace lattice::runtime::gpu

#endif  // LATTICE_RUNTIME_BACKENDS_GPU_CUBLAS_LOADER_H_
