#include <fstream>
#include <sstream>
#include <string>

#include "runtime/runner.h"
#include "test_util.h"

namespace test {

void RunInterpreterTests(TestContext* ctx) {
  auto env = std::make_shared<rt::Environment>();
  bt::InstallBuiltins(env);

  // Simple sequential statements.
  std::string src = "a = 2; b = a * 4; b + 1;";
  auto result = rt::RunSource(src, env);
  ExpectNear(Unwrap(result.value, "interpreter_sequence", ctx).number, 9.0, "interpreter_sequence",
             ctx);

  // Return stops execution.
  std::string src_return = "x = 1; return x + 5; x = 99;";
  auto ret = rt::RunSource(src_return, env);
  ExpectNear(Unwrap(ret.value, "interpreter_return", ctx).number, 6.0, "interpreter_return", ctx);

  // Functions inside script.
  std::string src_func =
      "func add(a, b) { return a + b; }\n"
      "func twice(x) { return add(x, x); }\n"
      "twice(7);";
  auto func_val = rt::RunSource(src_func, env);
  ExpectNear(Unwrap(func_val.value, "interpreter_functions", ctx).number, 14.0,
             "interpreter_functions", ctx);
}

}  // namespace test
