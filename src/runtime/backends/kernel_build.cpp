#include "runtime/backends/kernel_build.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace lattice::runtime {

namespace {

const char* BackendMacro(BackendType backend) {
    switch (backend) {
        case BackendType::kOpenCL:
            return "LATTICE_BACKEND_OPENCL";
        case BackendType::kCUDA:
            return "LATTICE_BACKEND_CUDA";
        case BackendType::kHIP:
            return "LATTICE_BACKEND_HIP";
        case BackendType::kMetal:
            return "LATTICE_BACKEND_METAL";
        case BackendType::kCPU:
            return "LATTICE_BACKEND_CPU";
    }
    return "LATTICE_BACKEND_CPU";
}

void AppendDefine(std::ostringstream& out,
                  const std::string& prefix,
                  const std::string& name,
                  const std::string& value) {
    if (!prefix.empty() && !name.empty()) {
        out << prefix << name;
        if (!value.empty()) {
            out << "=" << value;
        }
    }
}

std::string GetEnvString(const char* key) {
    if (!key)
        return "";
    const char* env = std::getenv(key);
    if (!env || env[0] == '\0')
        return "";
    return std::string(env);
}

bool IsTrueEnvValue(const char* value) {
    if (!value)
        return false;
    std::string v(value);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

constexpr std::array<uint32_t, 64> kSha256K = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

constexpr std::array<uint32_t, 8> kSha256Init = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

uint32_t Sha256RotateRight(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32 - bits));
}

void Sha256Transform(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        const size_t idx = static_cast<size_t>(i) * 4;
        w[i] = (static_cast<uint32_t>(block[idx]) << 24) |
               (static_cast<uint32_t>(block[idx + 1]) << 16) |
               (static_cast<uint32_t>(block[idx + 2]) << 8) |
               static_cast<uint32_t>(block[idx + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = Sha256RotateRight(w[i - 15], 7) ^
                      Sha256RotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = Sha256RotateRight(w[i - 2], 17) ^
                      Sha256RotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t s1 = Sha256RotateRight(e, 6) ^ Sha256RotateRight(e, 11) ^
                      Sha256RotateRight(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + s1 + ch + kSha256K[static_cast<size_t>(i)] + w[i];
        uint32_t s0 = Sha256RotateRight(a, 2) ^ Sha256RotateRight(a, 13) ^
                      Sha256RotateRight(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

struct Sha256State {
    uint32_t h[8];
    uint64_t total_bytes = 0;
    uint8_t buffer[64];
    size_t buffer_len = 0;
};

void Sha256Init(Sha256State* state) {
    std::memcpy(state->h, kSha256Init.data(), sizeof(state->h));
    state->total_bytes = 0;
    state->buffer_len = 0;
}

void Sha256Update(Sha256State* state, const uint8_t* data, size_t len) {
    if (!state || !data || len == 0)
        return;
    state->total_bytes += static_cast<uint64_t>(len);

    size_t offset = 0;
    if (state->buffer_len > 0) {
        size_t to_copy = std::min(len, 64 - state->buffer_len);
        std::memcpy(state->buffer + state->buffer_len, data, to_copy);
        state->buffer_len += to_copy;
        offset += to_copy;
        if (state->buffer_len == 64) {
            Sha256Transform(state->h, state->buffer);
            state->buffer_len = 0;
        }
    }

    while (offset + 64 <= len) {
        Sha256Transform(state->h, data + offset);
        offset += 64;
    }

    if (offset < len) {
        state->buffer_len = len - offset;
        std::memcpy(state->buffer, data + offset, state->buffer_len);
    }
}

void Sha256Final(Sha256State* state, uint8_t out[32]) {
    uint64_t total_bits = state->total_bytes * 8;
    state->buffer[state->buffer_len++] = 0x80;
    if (state->buffer_len > 56) {
        std::memset(state->buffer + state->buffer_len, 0,
                    64 - state->buffer_len);
        Sha256Transform(state->h, state->buffer);
        state->buffer_len = 0;
    }
    std::memset(state->buffer + state->buffer_len, 0, 56 - state->buffer_len);
    for (int i = 0; i < 8; ++i) {
        state->buffer[56 + i] =
            static_cast<uint8_t>((total_bits >> (56 - i * 8)) & 0xFF);
    }
    Sha256Transform(state->h, state->buffer);

    for (int i = 0; i < 8; ++i) {
        out[i * 4] = static_cast<uint8_t>((state->h[i] >> 24) & 0xFF);
        out[i * 4 + 1] = static_cast<uint8_t>((state->h[i] >> 16) & 0xFF);
        out[i * 4 + 2] = static_cast<uint8_t>((state->h[i] >> 8) & 0xFF);
        out[i * 4 + 3] = static_cast<uint8_t>(state->h[i] & 0xFF);
    }
}

}  // namespace

std::string KernelDefineString(const KernelBuildDefines& defs,
                               const std::string& prefix) {
    std::ostringstream out;
    bool first = true;
    auto add = [&](const std::string& name, const std::string& value = "") {
        if (!first)
            out << " ";
        first = false;
        AppendDefine(out, prefix, name, value);
    };

    add(BackendMacro(defs.backend), "1");
    if (defs.device_index >= 0) {
        add("LATTICE_DEVICE_INDEX", std::to_string(defs.device_index));
    }
    if (defs.device_type != 0) {
        add("LATTICE_DEVICE_TYPE", std::to_string(defs.device_type));
    }
    if (defs.vendor_id != 0) {
        add("LATTICE_VENDOR_ID", std::to_string(defs.vendor_id));
    }
    if (defs.abi_version != 0) {
        add("LATTICE_ABI_VERSION", std::to_string(defs.abi_version));
    }
    if (defs.abi_version_min != 0) {
        add("LATTICE_ABI_VERSION_MIN", std::to_string(defs.abi_version_min));
    }
    if (defs.has_fp16) {
        add("LATTICE_HAS_FP16", "1");
    }
    if (defs.has_fp64) {
        add("LATTICE_HAS_FP64", "1");
    }
    return out.str();
}

std::string LoadBuildOptionsEnv(const std::string& backend_env) {
    std::string global = GetEnvString("LATTICE_BUILD_OPTIONS");
    std::string backend = GetEnvString(backend_env.c_str());
    if (global.empty())
        return backend;
    if (backend.empty())
        return global;
    return global + " " + backend;
}

bool KernelDebugEnabled(const std::string& backend_env) {
    if (IsTrueEnvValue(std::getenv("LATTICE_BUILD_DEBUG"))) {
        return true;
    }
    if (!backend_env.empty()) {
        return IsTrueEnvValue(std::getenv(backend_env.c_str()));
    }
    return false;
}

bool KernelRebuildOnFailureEnabled(const std::string& backend_env) {
    if (!backend_env.empty()) {
        const char* backend = std::getenv(backend_env.c_str());
        if (backend && backend[0] != '\0')
            return IsTrueEnvValue(backend);
    }
    const char* global = std::getenv("LATTICE_REBUILD_ON_FAILURE");
    if (global && global[0] != '\0')
        return IsTrueEnvValue(global);
    return true;
}

std::string KernelDebugOptions(BackendType backend) {
    switch (backend) {
        case BackendType::kOpenCL:
            return "-g -cl-opt-disable";
        case BackendType::kCUDA:
            return "-G";
        case BackendType::kHIP:
            return "-g -O0";
        case BackendType::kMetal:
        case BackendType::kCPU:
            return "";
    }
    return "";
}

std::string Sha256Hex(const void* data, size_t size) {
    if (!data || size == 0)
        return "";
    Sha256State state;
    Sha256Init(&state);
    Sha256Update(&state, static_cast<const uint8_t*>(data), size);
    uint8_t digest[32];
    Sha256Final(&state, digest);
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.resize(64);
    for (size_t i = 0; i < 32; ++i) {
        out[i * 2] = kHex[(digest[i] >> 4) & 0xF];
        out[i * 2 + 1] = kHex[digest[i] & 0xF];
    }
    return out;
}

std::string Sha256Hex(const std::string& data) {
    return Sha256Hex(data.data(), data.size());
}

std::string NormalizePathArg(const std::string& path) {
    if (path.find(' ') == std::string::npos)
        return path;
    return "\"" + path + "\"";
}

std::string BuildIncludeOption(const std::string& include_dir,
                               const std::string& prefix) {
    if (include_dir.empty())
        return "";
    std::string normalized = NormalizePathArg(include_dir);
    if (prefix.empty())
        return normalized;
    if (!prefix.empty() && prefix.back() == ' ') {
        return prefix + normalized;
    }
    return prefix + " " + normalized;
}

void AppendOption(std::string* out, const std::string& option) {
    if (!out || option.empty())
        return;
    if (!out->empty())
        out->push_back(' ');
    out->append(option);
}

}  // namespace lattice::runtime
