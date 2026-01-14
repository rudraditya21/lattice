#include "runtime/backends/kernel_registry.h"

#include <array>

namespace lattice::runtime {

namespace {

constexpr std::array<KernelDefinition, 20> kKernelDefs = {{
    {"lattice_elemwise_add", "tensor_elemwise_add"},
    {"lattice_elemwise_sub", "tensor_elemwise_sub"},
    {"lattice_elemwise_mul", "tensor_elemwise_mul"},
    {"lattice_elemwise_div", "tensor_elemwise_div"},
    {"lattice_reduce_sum", "tensor_reduce_sum"},
    {"lattice_reduce_mean", "tensor_reduce_mean"},
    {"lattice_reduce_var", "tensor_reduce_var"},
    {"lattice_reduce_std", "tensor_reduce_std"},
    {"lattice_transpose", "tensor_transpose"},
    {"lattice_matmul", "tensor_matmul"},
    {"lattice_conv2d", "tensor_conv2d"},
    {"lattice_max_pool2d", "tensor_max_pool2d"},
    {"lattice_fft1d", "tensor_fft1d"},
    {"lattice_solve", "tensor_solve"},
    {"lattice_lu", "tensor_lu"},
    {"lattice_qr", "tensor_qr"},
    {"lattice_svd", "tensor_svd"},
    {"lattice_quantile", "tensor_quantile"},
    {"lattice_correlation", "tensor_correlation"},
    {"lattice_regression", "tensor_regression"},
}};

}  // namespace

const KernelDefinition* FindKernelDefinition(std::string_view name) {
    for (const auto& def : kKernelDefs) {
        if (def.name == name)
            return &def;
    }
    return nullptr;
}

std::vector<KernelDefinition> AllKernelDefinitions() {
    return std::vector<KernelDefinition>(kKernelDefs.begin(),
                                         kKernelDefs.end());
}

}  // namespace lattice::runtime
