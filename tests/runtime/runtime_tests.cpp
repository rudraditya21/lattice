#include "test_util.h"

namespace test {

void RunRuntimeTests(TestContext* ctx) {
  rt::Environment env;
  bt::InstallBuiltins(&env);

  ExpectNear(EvalExpr("1 + 2 * 3 - 4 / 2", &env).number, 5.0, "precedence", ctx);
  ExpectNear(EvalExpr("-3 + 5", &env).number, 2.0, "unary_minus", ctx);

  ExpectNear(EvalExpr("pow(2, 3)", &env).number, 8.0, "pow", ctx);
  ExpectNear(EvalExpr("gcd(12, 8)", &env).number, 4.0, "gcd", ctx);
  ExpectNear(EvalExpr("lcm(3, 5)", &env).number, 15.0, "lcm", ctx);
  ExpectNear(EvalExpr("abs(-3)", &env).number, 3.0, "abs", ctx);
  ExpectNear(EvalExpr("sign(-2)", &env).number, -1.0, "sign_neg", ctx);
  ExpectNear(EvalExpr("sign(0)", &env).number, 0.0, "sign_zero", ctx);
  ExpectNear(EvalExpr("sign(2)", &env).number, 1.0, "sign_pos", ctx);
  ExpectNear(EvalExpr("mod(10, 3)", &env).number, 1.0, "mod", ctx);
  ExpectNear(EvalExpr("floor(2.7)", &env).number, 2.0, "floor", ctx);
  ExpectNear(EvalExpr("ceil(2.2)", &env).number, 3.0, "ceil", ctx);
  ExpectNear(EvalExpr("round(2.5)", &env).number, 3.0, "round", ctx);
  ExpectNear(EvalExpr("clamp(5, 1, 4)", &env).number, 4.0, "clamp", ctx);
  ExpectNear(EvalExpr("min(2, 5)", &env).number, 2.0, "min", ctx);
  ExpectNear(EvalExpr("max(2, 5)", &env).number, 5.0, "max", ctx);

  bool caught = false;
  try {
    EvalExpr("mod(1, 0)", &env);
  } catch (const util::Error&) {
    caught = true;
  }
  ExpectTrue(caught, "mod_divide_by_zero_error", ctx);

  // Blocks and assignment.
  auto block_val = EvalStmt("{ x = 3; x + 2; }", &env);
  ExpectNear(block_val.value().number, 5.0, "block_assignment", ctx);

  // Conditional execution.
  auto if_val = EvalStmt("if (0) 10 else 7", &env);
  ExpectNear(if_val.value().number, 7.0, "if_else_execution", ctx);

  // Nested condition and guard that else is not evaluated when condition is true.
  auto nested = EvalStmt("if (1) { if (0) 9 else 8 } else 7", &env);
  ExpectNear(nested.value().number, 8.0, "nested_if", ctx);
  auto guard = EvalStmt("if (1) 5 else mod(1, 0)", &env);
  ExpectNear(guard.value().number, 5.0, "if_skips_else", ctx);
}

}  // namespace test
