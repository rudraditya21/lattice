#include <sstream>

#include "builtin/builtins.h"
#include "test_util.h"

namespace test {

void RunPrintTests(TestContext* ctx) {
  rt::Environment env;
  bt::InstallBuiltins(&env);
  bt::InstallPrint(&env);

  // We don't capture stdout here; just verify print returns 0 and doesn't throw.
  auto val = EvalExpr("print(5)", &env);
  ExpectNear(val.number, 0.0, "print_returns_zero", ctx);
}

}  // namespace test
