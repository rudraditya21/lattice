#include "test_util.h"

namespace test {

void RunParserTests(TestContext* ctx) {
  rt::Environment env;
  bt::InstallBuiltins(&env);

  ExpectNear(EvalExpr("(1 + 2) * 3", &env).number, 9.0, "grouping", ctx);

  bool caught = false;
  try {
    EvalExpr("1 2", &env);
  } catch (const util::Error&) {
    caught = true;
  }
  ExpectTrue(caught, "unexpected_token_error", ctx);
}

}  // namespace test
