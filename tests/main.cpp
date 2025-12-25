#include <iostream>

#include "test_util.h"

namespace test {
void RunLexerTests(TestContext* ctx);
void RunParserTests(TestContext* ctx);
void RunRuntimeTests(TestContext* ctx);
void RunBuiltinTests(TestContext* ctx);
void RunUtilTests(TestContext* ctx);
void RunReplTests(TestContext* ctx);
void RunInterpreterTests(TestContext* ctx);
void RunPrintTests(TestContext* ctx);
}  // namespace test

int main() {
  test::TestContext ctx;
  test::RunLexerTests(&ctx);
  test::RunParserTests(&ctx);
  test::RunRuntimeTests(&ctx);
  test::RunBuiltinTests(&ctx);
  test::RunUtilTests(&ctx);
  test::RunReplTests(&ctx);
  test::RunInterpreterTests(&ctx);
  test::RunPrintTests(&ctx);

  std::cout << "[RESULT] passed=" << ctx.passed << " failed=" << ctx.failed << "\n";
  return ctx.failed == 0 ? 0 : 1;
}
