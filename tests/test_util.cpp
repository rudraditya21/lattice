#include "test_util.h"

#include <iostream>

#include "runtime/type_checker.h"

namespace test {

void ExpectNear(double actual, double expected, const std::string& name, TestContext* ctx) {
  if (std::fabs(actual - expected) <= kEpsilon) {
    ++ctx->passed;
    return;
  }
  ++ctx->failed;
  std::cerr << "[FAIL] " << name << " expected " << expected << " got " << actual << "\n";
}

void ExpectTrue(bool value, const std::string& name, TestContext* ctx) {
  if (value) {
    ++ctx->passed;
    return;
  }
  ++ctx->failed;
  std::cerr << "[FAIL] " << name << " expected true\n";
}

rt::Value EvalExpr(const std::string& expr, rt::Environment* env) {
  lx::Lexer lex(expr);
  ps::Parser parser(std::move(lex));
  auto ast = parser.ParseExpression();
  rt::Evaluator evaluator(env);
  return evaluator.Evaluate(*ast);
}

rt::ExecResult EvalStmt(const std::string& stmt, rt::Environment* env) {
  lx::Lexer lex(stmt);
  ps::Parser parser(std::move(lex));
  auto parsed = parser.ParseStatement();
  rt::Evaluator evaluator(env);
  return evaluator.EvaluateStatement(*parsed);
}

void TypeCheckStmt(const std::string& stmt) {
  lx::Lexer lex(stmt);
  ps::Parser parser(std::move(lex));
  auto parsed = parser.ParseStatement();
  rt::TypeChecker checker;
  checker.Check(parsed.get());
}

const rt::Value& Unwrap(const std::optional<rt::Value>& v, const std::string& name,
                        TestContext* ctx) {
  ExpectTrue(v.has_value(), name + "_present", ctx);
  static rt::Value empty{};
  return v.has_value() ? v.value() : empty;
}

}  // namespace test
