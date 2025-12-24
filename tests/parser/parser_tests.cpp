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

  // If/else statement parses and evaluates.
  bool parsed_if = false;
  try {
    auto val = EvalStmt("if (1) 2 else 3", &env);
    parsed_if = true;
    ExpectNear(val.value().number, 2.0, "if_else_true_branch", ctx);
  } catch (const util::Error&) {
  }
  ExpectTrue(parsed_if, "if_else_parsed", ctx);
}

}  // namespace test
