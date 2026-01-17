#include "test_util.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "runtime/backends/cuda_backend.h"
#include "runtime/backends/hip_backend.h"
#include "runtime/backends/opencl_backend.h"
#if defined(__APPLE__)
#include "runtime/backends/metal_backend.h"
#endif

namespace test {

namespace {

bool EnvEnabled(const char* name) {
  const char* value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return false;
  }
  std::string v(value);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

template <typename T>
bool VectorsEqual(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace

void RunDeviceBufferTests(TestContext* ctx) {
  if (!EnvEnabled("LATTICE_GPU_TESTS")) {
    return;
  }

  {
    const auto* backend = rt::GetBackendByType(rt::BackendType::kOpenCL);
    if (rt::BackendAvailable(backend)) {
      const auto* opencl = static_cast<const rt::OpenCLBackend*>(backend);
      const int count = opencl->DeviceCount();
      const auto info = opencl->DeviceInfo();
      const auto caps = opencl->DeviceCaps();
      ExpectTrue(count == static_cast<int>(info.size()), "opencl_device_count", ctx);
      ExpectTrue(info.size() == caps.size(), "opencl_device_caps", ctx);
      if (count > 0) {
        ExpectTrue(!info[0].name.empty(), "opencl_device_name", ctx);
        std::vector<float> src(128);
        for (size_t i = 0; i < src.size(); ++i) {
          src[i] = static_cast<float>(i * 0.25f);
        }
        std::vector<float> dst(src.size(), 0.0f);
        auto buf_or = opencl->CreateBuffer(0, src.size() * sizeof(float));
        ExpectTrue(buf_or.ok(), "opencl_buffer_create", ctx);
        if (buf_or.ok()) {
          auto buf = buf_or.value();
          auto w = opencl->WriteBuffer(0, buf, src.data(), src.size() * sizeof(float));
          ExpectTrue(w.ok(), "opencl_buffer_write", ctx);
          auto r = opencl->ReadBuffer(0, buf, dst.data(), dst.size() * sizeof(float));
          ExpectTrue(r.ok(), "opencl_buffer_read", ctx);
          ExpectTrue(VectorsEqual(src, dst), "opencl_buffer_roundtrip", ctx);
          auto rel = opencl->ReleaseBuffer(&buf);
          ExpectTrue(rel.ok(), "opencl_buffer_release", ctx);
        }
      }
    }
  }

  {
    const auto* backend = rt::GetBackendByType(rt::BackendType::kCUDA);
    if (rt::BackendAvailable(backend)) {
      const auto* cuda = static_cast<const rt::CudaBackend*>(backend);
      const int count = cuda->DeviceCount();
      const auto info = cuda->DeviceInfo();
      const auto caps = cuda->DeviceCaps();
      ExpectTrue(count == static_cast<int>(info.size()), "cuda_device_count", ctx);
      ExpectTrue(info.size() == caps.size(), "cuda_device_caps", ctx);
      if (count > 0) {
        ExpectTrue(!info[0].name.empty(), "cuda_device_name", ctx);
        std::vector<float> src(128);
        for (size_t i = 0; i < src.size(); ++i) {
          src[i] = static_cast<float>(i * 0.5f);
        }
        std::vector<float> dst(src.size(), 0.0f);
        auto buf_or = cuda->CreateBuffer(0, src.size() * sizeof(float));
        ExpectTrue(buf_or.ok(), "cuda_buffer_create", ctx);
        if (buf_or.ok()) {
          auto buf = buf_or.value();
          auto w = cuda->WriteBuffer(0, buf, src.data(), src.size() * sizeof(float));
          ExpectTrue(w.ok(), "cuda_buffer_write", ctx);
          auto r = cuda->ReadBuffer(0, buf, dst.data(), dst.size() * sizeof(float));
          ExpectTrue(r.ok(), "cuda_buffer_read", ctx);
          ExpectTrue(VectorsEqual(src, dst), "cuda_buffer_roundtrip", ctx);
          auto rel = cuda->ReleaseBuffer(&buf);
          ExpectTrue(rel.ok(), "cuda_buffer_release", ctx);
        }
      }
    }
  }

  {
    const auto* backend = rt::GetBackendByType(rt::BackendType::kHIP);
    if (rt::BackendAvailable(backend)) {
      const auto* hip = static_cast<const rt::HipBackend*>(backend);
      const int count = hip->DeviceCount();
      const auto info = hip->DeviceInfo();
      const auto caps = hip->DeviceCaps();
      ExpectTrue(count == static_cast<int>(info.size()), "hip_device_count", ctx);
      ExpectTrue(info.size() == caps.size(), "hip_device_caps", ctx);
      if (count > 0) {
        ExpectTrue(!info[0].name.empty(), "hip_device_name", ctx);
        std::vector<float> src(128);
        for (size_t i = 0; i < src.size(); ++i) {
          src[i] = static_cast<float>(i * 0.75f);
        }
        std::vector<float> dst(src.size(), 0.0f);
        auto buf_or = hip->CreateBuffer(0, src.size() * sizeof(float));
        ExpectTrue(buf_or.ok(), "hip_buffer_create", ctx);
        if (buf_or.ok()) {
          auto buf = buf_or.value();
          auto w = hip->WriteBuffer(0, buf, src.data(), src.size() * sizeof(float));
          ExpectTrue(w.ok(), "hip_buffer_write", ctx);
          auto r = hip->ReadBuffer(0, buf, dst.data(), dst.size() * sizeof(float));
          ExpectTrue(r.ok(), "hip_buffer_read", ctx);
          ExpectTrue(VectorsEqual(src, dst), "hip_buffer_roundtrip", ctx);
          auto rel = hip->ReleaseBuffer(&buf);
          ExpectTrue(rel.ok(), "hip_buffer_release", ctx);
        }
      }
    }
  }

#if defined(__APPLE__)
  {
    const auto* backend = rt::GetBackendByType(rt::BackendType::kMetal);
    if (rt::BackendAvailable(backend)) {
      const auto* metal = static_cast<const rt::MetalBackend*>(backend);
      const int count = metal->DeviceCount();
      const auto info = metal->DeviceInfo();
      const auto caps = metal->DeviceCaps();
      ExpectTrue(count == static_cast<int>(info.size()), "metal_device_count", ctx);
      ExpectTrue(info.size() == caps.size(), "metal_device_caps", ctx);
      if (count > 0) {
        ExpectTrue(!info[0].name.empty(), "metal_device_name", ctx);
        std::vector<float> src(128);
        for (size_t i = 0; i < src.size(); ++i) {
          src[i] = static_cast<float>(i);
        }
        std::vector<float> dst(src.size(), 0.0f);
        auto buf_or = metal->CreateBuffer(0, src.size() * sizeof(float));
        ExpectTrue(buf_or.ok(), "metal_buffer_create", ctx);
        if (buf_or.ok()) {
          auto buf = buf_or.value();
          auto w = metal->WriteBuffer(0, buf, src.data(), src.size() * sizeof(float));
          ExpectTrue(w.ok(), "metal_buffer_write", ctx);
          auto r = metal->ReadBuffer(0, buf, dst.data(), dst.size() * sizeof(float));
          ExpectTrue(r.ok(), "metal_buffer_read", ctx);
          ExpectTrue(VectorsEqual(src, dst), "metal_buffer_roundtrip", ctx);
          auto rel = metal->ReleaseBuffer(&buf);
          ExpectTrue(rel.ok(), "metal_buffer_release", ctx);
        }
      }
    }
  }
#endif
}

}  // namespace test
