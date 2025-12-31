#ifndef LATTICE_TESTS_TEST_UTIL_H_
#define LATTICE_TESTS_TEST_UTIL_H_

#include <cmath>
#include <memory>
#include <optional>
#include <string>

#include "builtin/builtins.h"
#include "lexer/lexer.h"
#include "parser/parser.h"
#include "runtime/environment.h"
#include "runtime/ops.h"
#include "util/error.h"

namespace test {

namespace rt = lattice::runtime;
namespace lx = lattice::lexer;
namespace ps = lattice::parser;
namespace bt = lattice::builtin;
namespace util = lattice::util;

struct TestContext {
  int passed = 0;
  int failed = 0;
};

constexpr double kEpsilon = 1e-9;

void ExpectNear(double actual, double expected, const std::string& name, TestContext* ctx);
void ExpectTrue(bool value, const std::string& name, TestContext* ctx);

rt::Value EvalExpr(const std::string& expr, std::shared_ptr<rt::Environment> env);
rt::ExecResult EvalStmt(const std::string& stmt, std::shared_ptr<rt::Environment> env);
void TypeCheckStmt(const std::string& stmt);
// Helper to unwrap optional values in tests with a presence check.
const rt::Value& Unwrap(const std::optional<rt::Value>& v, const std::string& name,
                        TestContext* ctx);

}  // namespace test

#endif  // LATTICE_TESTS_TEST_UTIL_H_
