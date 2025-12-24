#ifndef LATTICE_TESTS_TEST_UTIL_H_
#define LATTICE_TESTS_TEST_UTIL_H_

#include <cmath>
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

rt::Value EvalExpr(const std::string& expr, rt::Environment* env);
std::optional<rt::Value> EvalStmt(const std::string& stmt, rt::Environment* env);

}  // namespace test

#endif  // LATTICE_TESTS_TEST_UTIL_H_
