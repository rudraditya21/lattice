#include "test_util.h"

#include <iostream>

#include "runtime/runner.h"

namespace test {

namespace {

rt::ExecResult RunProgram(const std::string& src, const std::shared_ptr<rt::Environment>& env) {
  bt::InstallBuiltins(env);
  return rt::RunSource(src, env);
}

bool TryNumberResult(const std::string& src, double* out, std::string* error) {
  auto env = std::make_shared<rt::Environment>();
  try {
    auto result = RunProgram(src, env);
    if (!result.value.has_value()) {
      if (error) *error = "no value returned";
      return false;
    }
    if (out) *out = result.value->f64;
    return true;
  } catch (const util::Error& e) {
    if (error) *error = e.formatted();
    return false;
  }
}

}  // namespace

void RunIntegrationTests(TestContext* ctx) {
  {
    std::string program =
        "func fact(n) { if (n <= 1) { return 1; } else { return n * fact(n - 1); } }\n"
        "x = fact(5);\n"
        "t = tensor_values(((1,2),(3,4)));\n"
        "u = matmul(t, t);\n"
        "s = sum(u);\n"
        "r = {\"x\": x, \"s\": s};\n"
        "r[\"x\"] + s;";
    double result = 0.0;
    std::string error;
    bool ok = TryNumberResult(program, &result, &error);
    ExpectTrue(ok, "integration_fact_matmul_sum_ok", ctx);
    if (!ok) {
      std::cerr << "[FAIL] integration_fact_matmul_sum_error " << error << "\n";
    } else {
      ExpectNear(result, 174.0, "integration_fact_matmul_sum_value", ctx);
    }
  }

  {
    std::string program = "(a, b) = (3, 4); r = {y: 5}; a * b + r[\"y\"];";
    double result = 0.0;
    std::string error;
    bool ok = TryNumberResult(program, &result, &error);
    ExpectTrue(ok, "integration_destructure_ok", ctx);
    if (!ok) {
      std::cerr << "[FAIL] integration_destructure_error " << error << "\n";
    } else {
      ExpectNear(result, 17.0, "integration_destructure_value", ctx);
    }
  }

  {
    std::string program =
        "sum = 0;\n"
        "for (i = 0; i < 5; i = i + 1) { if (i == 2) { continue; } sum = sum + i; };\n"
        "sum;";
    double result = 0.0;
    std::string error;
    bool ok = TryNumberResult(program, &result, &error);
    ExpectTrue(ok, "integration_for_continue_ok", ctx);
    if (!ok) {
      std::cerr << "[FAIL] integration_for_continue_error " << error << "\n";
    } else {
      ExpectNear(result, 8.0, "integration_for_continue_value", ctx);
    }
  }

  {
    bool caught = false;
    auto env = std::make_shared<rt::Environment>();
    try {
      RunProgram("x: i32 = 2.5;", env);
    } catch (const util::Error&) {
      caught = true;
    }
    ExpectTrue(caught, "integration_type_error", ctx);
  }
}

}  // namespace test
