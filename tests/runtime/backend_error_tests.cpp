#include "test_util.h"

#include "runtime/backends/backend_error.h"

namespace test {

void RunBackendErrorTests(TestContext* ctx) {
  auto status =
      rt::MakeBackendError(rt::StatusCode::kInternal, rt::BackendType::kOpenCL,
                           rt::BackendErrorKind::kBuild, "compile failed");
  ExpectTrue(status.code == rt::StatusCode::kInternal, "backend_error_code", ctx);
  ExpectTrue(status.message.find("[opencl][build]") != std::string::npos,
             "backend_error_prefix", ctx);

  ExpectTrue(std::string(rt::BackendTypeName(rt::BackendType::kCUDA)) == "cuda",
             "backend_type_name", ctx);
  ExpectTrue(std::string(rt::BackendErrorKindName(rt::BackendErrorKind::kLaunch)) == "launch",
             "backend_kind_name", ctx);
}

}  // namespace test
