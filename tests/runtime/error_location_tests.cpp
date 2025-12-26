#include "test_util.h"

namespace test {

void RunErrorLocationTests(TestContext* ctx) {
  rt::Environment env;
  bt::InstallBuiltins(&env);

  auto expect_loc = [&](const std::string& code, const std::string& name) {
    bool has_loc = false;
    try {
      EvalExpr(code, &env);
    } catch (const util::Error& e) {
      has_loc = e.line() > 0 && e.column() > 0;
    }
    ExpectTrue(has_loc, name, ctx);
  };

  expect_loc("\"a\" + 1", "string_plus_number_location");
  expect_loc("gcd(decimal(1.0), 2)", "gcd_type_error_location");
  expect_loc("1 / 0", "divide_by_zero_location");
  expect_loc("unknown_fn(1)", "unknown_function_location");
}

}  // namespace test
