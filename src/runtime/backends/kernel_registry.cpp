#include "runtime/backends/kernel_registry.h"

#include <array>
#include <cstdint>

namespace lattice::runtime {

namespace {

struct KernelDispatchEntry {
    KernelOp op;
    BackendType backend;
    KernelVendor vendor;
    uint32_t min_arch_major;
    uint32_t max_arch_major;
    bool requires_vectorize;
    bool requires_fp64;
    std::string_view kernel_name;
};

constexpr std::array<KernelDefinition, 28> kKernelDefs = {{
    {"lattice_elemwise_add", "tensor_elemwise_add"},
    {"lattice_elemwise_add_vec4", "tensor_elemwise_add"},
    {"lattice_elemwise_sub", "tensor_elemwise_sub"},
    {"lattice_elemwise_sub_vec4", "tensor_elemwise_sub"},
    {"lattice_elemwise_mul", "tensor_elemwise_mul"},
    {"lattice_elemwise_mul_vec4", "tensor_elemwise_mul"},
    {"lattice_elemwise_div", "tensor_elemwise_div"},
    {"lattice_elemwise_div_vec4", "tensor_elemwise_div"},
    {"lattice_reduce_sum", "tensor_reduce_sum"},
    {"lattice_reduce_mean", "tensor_reduce_mean"},
    {"lattice_reduce_var", "tensor_reduce_var"},
    {"lattice_reduce_std", "tensor_reduce_std"},
    {"lattice_transpose", "tensor_transpose"},
    {"lattice_matmul", "tensor_matmul"},
    {"lattice_matmul_t16", "tensor_matmul"},
    {"lattice_matmul_t32", "tensor_matmul"},
    {"lattice_conv2d", "tensor_conv2d"},
    {"lattice_conv2d_t8", "tensor_conv2d"},
    {"lattice_conv2d_t16", "tensor_conv2d"},
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

constexpr std::array<KernelDispatchEntry, 25> kDispatchTable = {{
    {KernelOp::kElemwiseAdd, BackendType::kCPU, KernelVendor::kAny, 0, 0, true,
     false, "lattice_elemwise_add_vec4"},
    {KernelOp::kElemwiseAdd, BackendType::kCPU, KernelVendor::kAny, 0, 0, false,
     false, "lattice_elemwise_add"},
    {KernelOp::kElemwiseSub, BackendType::kCPU, KernelVendor::kAny, 0, 0, true,
     false, "lattice_elemwise_sub_vec4"},
    {KernelOp::kElemwiseSub, BackendType::kCPU, KernelVendor::kAny, 0, 0, false,
     false, "lattice_elemwise_sub"},
    {KernelOp::kElemwiseMul, BackendType::kCPU, KernelVendor::kAny, 0, 0, true,
     false, "lattice_elemwise_mul_vec4"},
    {KernelOp::kElemwiseMul, BackendType::kCPU, KernelVendor::kAny, 0, 0, false,
     false, "lattice_elemwise_mul"},
    {KernelOp::kElemwiseDiv, BackendType::kCPU, KernelVendor::kAny, 0, 0, true,
     false, "lattice_elemwise_div_vec4"},
    {KernelOp::kElemwiseDiv, BackendType::kCPU, KernelVendor::kAny, 0, 0, false,
     false, "lattice_elemwise_div"},

    {KernelOp::kReduceSum, BackendType::kCPU, KernelVendor::kAny, 0, 0, false,
     false, "lattice_reduce_sum"},
    {KernelOp::kReduceMean, BackendType::kCPU, KernelVendor::kAny, 0, 0, false,
     false, "lattice_reduce_mean"},
    {KernelOp::kReduceVar, BackendType::kCPU, KernelVendor::kAny, 0, 0, false,
     false, "lattice_reduce_var"},
    {KernelOp::kReduceStd, BackendType::kCPU, KernelVendor::kAny, 0, 0, false,
     false, "lattice_reduce_std"},

    {KernelOp::kTranspose, BackendType::kCPU, KernelVendor::kAny, 0, 0, false,
     false, "lattice_transpose"},

    {KernelOp::kMatmul, BackendType::kCUDA, KernelVendor::kNvidia, 8, 0, false,
     false, "lattice_matmul_t32"},
    {KernelOp::kMatmul, BackendType::kCUDA, KernelVendor::kNvidia, 0, 0, false,
     false, "lattice_matmul_t16"},
    {KernelOp::kMatmul, BackendType::kHIP, KernelVendor::kAmd, 0, 0, false,
     false, "lattice_matmul_t16"},
    {KernelOp::kMatmul, BackendType::kMetal, KernelVendor::kApple, 0, 0, false,
     false, "lattice_matmul_t16"},
    {KernelOp::kMatmul, BackendType::kOpenCL, KernelVendor::kIntel, 0, 0, false,
     false, "lattice_matmul_t16"},
    {KernelOp::kMatmul, BackendType::kCPU, KernelVendor::kAny, 0, 0, false,
     false, "lattice_matmul"},

    {KernelOp::kConv2d, BackendType::kCUDA, KernelVendor::kNvidia, 0, 0, false,
     false, "lattice_conv2d_t16"},
    {KernelOp::kConv2d, BackendType::kHIP, KernelVendor::kAmd, 0, 0, false,
     false, "lattice_conv2d_t16"},
    {KernelOp::kConv2d, BackendType::kMetal, KernelVendor::kApple, 0, 0, false,
     false, "lattice_conv2d_t16"},
    {KernelOp::kConv2d, BackendType::kOpenCL, KernelVendor::kIntel, 0, 0, false,
     false, "lattice_conv2d_t8"},
    {KernelOp::kConv2d, BackendType::kOpenCL, KernelVendor::kAny, 0, 0, false,
     false, "lattice_conv2d_t16"},
    {KernelOp::kConv2d, BackendType::kCPU, KernelVendor::kAny, 0, 0, false,
     false, "lattice_conv2d_t16"},
}};

bool MatchDispatch(const KernelDispatchEntry& entry,
                   const KernelDispatchKey& key) {
    if (entry.backend != BackendType::kCPU && entry.backend != key.backend) {
        return false;
    }
    if (entry.vendor != KernelVendor::kAny && entry.vendor != key.vendor) {
        return false;
    }
    if (entry.requires_vectorize && (!key.vectorize || key.use_fp64)) {
        return false;
    }
    if (entry.requires_fp64 && !key.use_fp64) {
        return false;
    }
    if (entry.min_arch_major != 0 && key.arch_major < entry.min_arch_major) {
        return false;
    }
    if (entry.max_arch_major != 0 && key.arch_major > entry.max_arch_major) {
        return false;
    }
    return true;
}

int DispatchScore(const KernelDispatchEntry& entry,
                  const KernelDispatchKey& key) {
    int score = 0;
    if (entry.backend == key.backend) {
        score += 4;
    } else if (entry.backend == BackendType::kCPU) {
        score += 1;
    }
    if (entry.vendor == key.vendor) {
        score += 2;
    } else if (entry.vendor == KernelVendor::kAny) {
        score += 1;
    }
    if (entry.min_arch_major != 0 || entry.max_arch_major != 0) {
        score += 1;
    }
    if (entry.requires_vectorize) {
        score += 1;
    }
    if (entry.requires_fp64) {
        score += 1;
    }
    return score;
}

}  // namespace

const KernelDefinition* FindKernelDefinition(std::string_view name) {
    for (const auto& def : kKernelDefs) {
        if (def.name == name)
            return &def;
    }
    return nullptr;
}

const KernelDefinition* SelectKernelDefinition(KernelOp op,
                                               const KernelDispatchKey& key) {
    const KernelDispatchEntry* best = nullptr;
    int best_score = -1;
    for (const auto& entry : kDispatchTable) {
        if (entry.op != op)
            continue;
        if (!MatchDispatch(entry, key))
            continue;
        int score = DispatchScore(entry, key);
        if (score > best_score) {
            best_score = score;
            best = &entry;
        }
    }
    if (!best)
        return nullptr;
    return FindKernelDefinition(best->kernel_name);
}

std::vector<KernelDefinition> AllKernelDefinitions() {
    return std::vector<KernelDefinition>(kKernelDefs.begin(),
                                         kKernelDefs.end());
}

}  // namespace lattice::runtime
