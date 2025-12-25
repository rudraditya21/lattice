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
  ExpectTrue(EvalExpr("int(2.9)", &env).type == rt::DType::kI64, "int_cast_type", ctx);
  ExpectTrue(EvalExpr("float(2)", &env).type == rt::DType::kF64, "float_cast_type", ctx);

  // Overload/error paths for math builtins.
  bool gcd_rejects_decimal = false;
  try {
    EvalExpr("gcd(decimal(1.0), 2)", &env);
  } catch (const util::Error&) {
    gcd_rejects_decimal = true;
  }
  ExpectTrue(gcd_rejects_decimal, "gcd_rejects_decimal", ctx);

  auto abs_cmplx = EvalExpr("abs(complex(3, 4))", &env);
  ExpectNear(abs_cmplx.f64, 5.0, "abs_complex_magnitude", ctx);

  bool mod_zero = false;
  try {
    EvalExpr("mod(1, 0)", &env);
  } catch (const util::Error&) {
    mod_zero = true;
  }
  ExpectTrue(mod_zero, "mod_zero_errors", ctx);
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
  ExpectNear(block_val.value.value().number, 5.0, "block_assignment", ctx);

  // Conditional execution.
  auto if_val = EvalStmt("if (0) 10 else 7", &env);
  ExpectNear(if_val.value.value().number, 7.0, "if_else_execution", ctx);

  // Nested condition and guard that else is not evaluated when condition is true.
  auto nested = EvalStmt("if (1) { if (0) 9 else 8 } else 7", &env);
  ExpectNear(nested.value.value().number, 8.0, "nested_if", ctx);
  auto guard = EvalStmt("if (1) 5 else mod(1, 0)", &env);
  ExpectNear(guard.value.value().number, 5.0, "if_skips_else", ctx);

  // While loops.
  auto loop = EvalStmt("{ i = 0; while (i - 3) { i = i + 1; }; i }", &env);
  ExpectNear(loop.value.value().number, 3.0, "while_basic_increment", ctx);
  auto zero_loop = EvalStmt("{ j = 0; while (0) { j = j + 1; }; j }", &env);
  ExpectNear(zero_loop.value.value().number, 0.0, "while_does_not_run_on_false", ctx);

  // Break/continue in while.
  auto with_break = EvalStmt("{ k = 0; while (1) { k = k + 1; if (k - 2) { } else { break; } }; k }", &env);
  ExpectNear(with_break.value.value().number, 2.0, "while_break", ctx);
  auto with_continue = EvalStmt("{ s = 0; t = 0; while (s - 3) { s = s + 1; if (s - 2) { } else { continue; } t = t + 1; }; t }", &env);
  ExpectNear(with_continue.value.value().number, 2.0, "while_continue", ctx);

  // For loops.
  auto for_sum = EvalStmt("{ acc = 0; for (i = 0; i - 4; i = i + 1) { acc = acc + i; }; acc }", &env);
  ExpectNear(for_sum.value.value().number, 6.0, "for_loop_sum", ctx);
  auto for_no_cond = EvalStmt("{ n = 0; for (; n - 2; ) { n = n + 1; }; n }", &env);
  ExpectNear(for_no_cond.value.value().number, 2.0, "for_loop_no_cond", ctx);
  auto for_break = EvalStmt("{ x = 0; for (i = 0; i - 5; i = i + 1) { if (i - 3) { } else { break; } x = x + 1; }; x }", &env);
  ExpectNear(for_break.value.value().number, 3.0, "for_break", ctx);

  // Equality/inequality and comparisons.
  ExpectNear(EvalExpr("1 == 1", &env).number, 1.0, "eq_true", ctx);
  ExpectNear(EvalExpr("1 == 2", &env).number, 0.0, "eq_false", ctx);
  ExpectNear(EvalExpr("3 != 2", &env).number, 1.0, "ne_true", ctx);
  ExpectNear(EvalExpr("3 != 3", &env).number, 0.0, "ne_false", ctx);
  ExpectNear(EvalExpr("3 > 2", &env).number, 1.0, "gt_true", ctx);
  ExpectNear(EvalExpr("2 > 3", &env).number, 0.0, "gt_false", ctx);
  ExpectNear(EvalExpr("3 >= 3", &env).number, 1.0, "ge_true", ctx);
  ExpectNear(EvalExpr("2 < 5", &env).number, 1.0, "lt_true", ctx);
  ExpectNear(EvalExpr("5 <= 2", &env).number, 0.0, "le_false", ctx);
  auto if_eq = EvalStmt("if (2 == 2) 9 else 1", &env);
  ExpectNear(if_eq.value.value().number, 9.0, "if_with_equality", ctx);

  // Boolean literals and coercion in control flow.
  auto bool_true = EvalExpr("true", &env);
  ExpectTrue(bool_true.boolean, "bool_literal_true", ctx);
  auto bool_false = EvalExpr("false", &env);
  ExpectTrue(!bool_false.boolean, "bool_literal_false", ctx);
  auto if_bool = EvalStmt("if (false) 1 else 2", &env);
  ExpectNear(if_bool.value.value().number, 2.0, "if_with_false_literal", ctx);

  // Functions.
  auto func_simple =
      EvalStmt("{ func add(a, b) { return a + b; } add(2, 3); }", &env);
  ExpectNear(func_simple.value.value().number, 5.0, "func_add", ctx);

  auto func_no_return = EvalStmt("{ func inc(x) { x = x + 1; } inc(4); }", &env);
  ExpectNear(func_no_return.value.value().number, 5.0, "func_returns_last_value_when_no_return",
             ctx);

  auto func_nested = EvalStmt(
      "{ func fact(n) { if (n <= 1) { return 1; } else { return n * fact(n - 1); } } fact(5); }",
      &env);
  ExpectNear(func_nested.value.value().number, 120.0, "func_recursion", ctx);

  // Type annotations enforcement.
  bool caught_assign = false;
  try {
    EvalStmt("x: bool = 3", &env);
  } catch (const util::Error&) {
    caught_assign = true;
  }
  ExpectTrue(caught_assign, "annotated_assign_mismatch", ctx);

  bool caught_param = false;
  try {
    EvalStmt("{ func addi(a: i32) { return a; } addi(true); }", &env);
  } catch (const util::Error&) {
    caught_param = true;
  }
  ExpectTrue(caught_param, "annotated_param_mismatch", ctx);

  bool caught_ret = false;
  try {
    EvalStmt("{ func give_bool(): bool { return 3; } give_bool(); }", &env);
  } catch (const util::Error&) {
    caught_ret = true;
  }
  ExpectTrue(caught_ret, "annotated_return_mismatch", ctx);

  // Type metadata propagates.
  auto typed_val = EvalStmt("{ y: i32 = 5; y + 1; }", &env);
  auto tn = typed_val.value.value().type_name;
  ExpectTrue(tn == "i32" || tn == "i64", "typed_value_annotation", ctx);
  auto typed_ret = EvalStmt("{ func addi(a: i32, b: i32) -> i32 { return a + b; } addi(1, 2); }",
                            &env);
  auto tn2 = typed_ret.value.value().type_name;
  ExpectTrue(tn2 == "i32" || tn2 == "i64", "typed_return_annotation", ctx);
}

}  // namespace test
