#include "test_util.h"

namespace test {

void RunParserTests(TestContext* ctx) {
  auto env = std::make_shared<rt::Environment>();
  bt::InstallBuiltins(env);
  auto expect_error = [&](const std::string& src, const std::string& name) {
    bool caught = false;
    try {
      lx::Lexer lex(src);
      ps::Parser parser(std::move(lex));
      parser.ParseStatement();
    } catch (const util::Error&) {
      caught = true;
    }
    ExpectTrue(caught, name, ctx);
  };

  ExpectNear(EvalExpr("(1 + 2) * 3", env).number, 9.0, "grouping", ctx);

  bool caught = false;
  try {
    EvalExpr("1 2", env);
  } catch (const util::Error&) {
    caught = true;
  }
  ExpectTrue(caught, "unexpected_token_error", ctx);

  // If/else statement parses and evaluates.
  bool parsed_if = false;
  try {
    auto val = EvalStmt("if (true) 2 else 3", env);
    parsed_if = true;
    ExpectNear(Unwrap(val.value, "if_else_true_branch", ctx).number, 2.0, "if_else_true_branch",
               ctx);
  } catch (const util::Error&) {
  }
  ExpectTrue(parsed_if, "if_else_parsed", ctx);

  expect_error("if (true 1 else 2", "if_missing_paren_error");
  expect_error("func f(a b) { return a; }", "func_missing_comma_error");
  expect_error("for (i = 0 i < 3; i = i + 1) { }", "for_missing_semicolon_error");

  try {
    lx::Lexer lex("return;");
    ps::Parser parser(std::move(lex));
    auto stmt = parser.ParseStatement();
    auto* ret = dynamic_cast<ps::ReturnStatement*>(stmt.get());
    ExpectTrue(ret != nullptr && ret->expr == nullptr, "return_without_expr", ctx);
  } catch (const util::Error&) {
    ExpectTrue(false, "return_without_expr_parse", ctx);
  }

  // Type-annotated function and variable parse and enforce.
  bool parsed_types = true;
  try {
    EvalStmt("func add(a: i32, b: f64) -> f64 { return a + b; }", env);
    EvalStmt("x: i32 = 3", env);
  } catch (const util::Error&) {
    parsed_types = false;
  }
  ExpectTrue(parsed_types, "annotated_syntax_parses", ctx);

  // DType resolution on annotations.
  try {
    lx::Lexer lex("func add(a: i32, b: f64) -> f64 { return a + b; }");
    ps::Parser p(std::move(lex));
    auto stmt = p.ParseStatement();
    auto* fn = dynamic_cast<ps::FunctionStatement*>(stmt.get());
    ExpectTrue(fn != nullptr, "function_statement_parsed", ctx);
    auto dt_a = fn->parameter_types[0].type->dtype;
    auto dt_b = fn->parameter_types[1].type->dtype;
    auto dt_ret = fn->return_type.type->dtype;
    ExpectTrue(dt_a.has_value() && dt_a.value() == rt::DType::kI32, "dtype_param_a_i32", ctx);
    ExpectTrue(
        dt_b.has_value() && (dt_b.value() == rt::DType::kF32 || dt_b.value() == rt::DType::kF64),
        "dtype_param_b_float", ctx);
    ExpectTrue(dt_ret.has_value() &&
                   (dt_ret.value() == rt::DType::kF32 || dt_ret.value() == rt::DType::kF64),
               "dtype_return_float", ctx);
  } catch (const util::Error&) {
    ExpectTrue(false, "dtype_resolution_failed", ctx);
  }
}

}  // namespace test
