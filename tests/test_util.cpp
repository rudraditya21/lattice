#include "test_util.h"

#include <cstdlib>
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

ScopedEnvVar::ScopedEnvVar(const std::string& name, const std::string& value) : name_(name) {
  const char* current = std::getenv(name_.c_str());
  if (current) {
    had_value_ = true;
    old_value_ = current;
  }
  SetEnv(value);
}

ScopedEnvVar::~ScopedEnvVar() {
  if (had_value_) {
    SetEnv(old_value_);
  } else {
    UnsetEnv();
  }
}

void ScopedEnvVar::SetEnv(const std::string& value) {
#if defined(_WIN32)
  _putenv_s(name_.c_str(), value.c_str());
#else
  setenv(name_.c_str(), value.c_str(), 1);
#endif
}

void ScopedEnvVar::UnsetEnv() {
#if defined(_WIN32)
  _putenv_s(name_.c_str(), "");
#else
  unsetenv(name_.c_str());
#endif
}

rt::Value EvalExpr(const std::string& expr, std::shared_ptr<rt::Environment> env) {
  lx::Lexer lex(expr);
  ps::Parser parser(std::move(lex));
  auto ast = parser.ParseExpression();
  rt::Evaluator evaluator(std::move(env));
  return evaluator.Evaluate(*ast);
}

rt::ExecResult EvalStmt(const std::string& stmt, std::shared_ptr<rt::Environment> env) {
  lx::Lexer lex(stmt);
  ps::Parser parser(std::move(lex));
  auto parsed = parser.ParseStatement();
  rt::Evaluator evaluator(std::move(env));
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
