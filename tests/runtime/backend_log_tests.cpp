#include "test_util.h"

#include <filesystem>
#include <string>

#include "runtime/backends/backend_log.h"

namespace test {

void RunBackendLogTests(TestContext* ctx) {
  rt::LogRecord rec;
  rec.level = rt::LogLevel::kInfo;
  rec.backend = rt::BackendType::kCUDA;
  rec.kind = rt::BackendErrorKind::kLaunch;
  rec.message = "launch ok";
  rec.operation = "launch";
  rec.device_index = 1;
  rec.device_name = "Test GPU";
  rec.error_code = 7;
  rec.error_name = "ERR";
  rec.trace_path = "/tmp/trace";

  std::string text = rt::FormatLogLine(rec, rt::LogFormat::kText);
  ExpectTrue(text.find("backend=cuda") != std::string::npos, "log_text_backend", ctx);
  ExpectTrue(text.find("trace_path") != std::string::npos, "log_text_trace", ctx);

  std::string json = rt::FormatLogLine(rec, rt::LogFormat::kJson);
  ExpectTrue(json.find("\"backend\":\"cuda\"") != std::string::npos, "log_json_backend", ctx);
  ExpectTrue(json.find("\"trace_path\"") != std::string::npos, "log_json_trace", ctx);

  const std::filesystem::path trace_root = MakeTempDir("lattice_trace_test_");
  ScopedEnvVar trace_env("LATTICE_TRACE_KERNELS", "1");
  ScopedEnvVar trace_dir("LATTICE_TRACE_DIR", trace_root.string());
  rt::KernelTrace trace;
  trace.backend = rt::BackendType::kOpenCL;
  trace.kernel_name = "vec_add";
  trace.build_options = "-cl-std=CL2.0";
  trace.source = "__kernel void vec_add() {}";
  trace.device_index = 0;
  trace.device_name = "Trace Device";

  std::string path;
  bool wrote = rt::TraceKernelSource(trace, &path);
  ExpectTrue(wrote, "trace_kernel_written", ctx);
  ExpectTrue(!path.empty() && std::filesystem::exists(path), "trace_kernel_file_exists", ctx);
}

}  // namespace test
