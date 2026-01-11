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
void RunPromotionTests(TestContext* ctx);
void RunTypeCheckerTests(TestContext* ctx);
void RunDecimalRationalTests(TestContext* ctx);
void RunComplexTests(TestContext* ctx);
void RunTensorTests(TestContext* ctx);
void RunErrorLocationTests(TestContext* ctx);
void RunBackendTests(TestContext* ctx);
void RunBackendEdgeTests(TestContext* ctx);
void RunBackendErrorTests(TestContext* ctx);
void RunBackendLogTests(TestContext* ctx);
void RunAbiTests(TestContext* ctx);
void RunDeviceSelectorTests(TestContext* ctx);
void RunDeviceQuirksTests(TestContext* ctx);
void RunCacheStoreTests(TestContext* ctx);
void RunMemoryPoolTests(TestContext* ctx);
void RunIntegrationTests(TestContext* ctx);
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
  test::RunPromotionTests(&ctx);
  test::RunTypeCheckerTests(&ctx);
  test::RunDecimalRationalTests(&ctx);
  test::RunComplexTests(&ctx);
  test::RunTensorTests(&ctx);
  test::RunErrorLocationTests(&ctx);
  test::RunBackendTests(&ctx);
  test::RunBackendEdgeTests(&ctx);
  test::RunBackendErrorTests(&ctx);
  test::RunBackendLogTests(&ctx);
  test::RunAbiTests(&ctx);
  test::RunDeviceSelectorTests(&ctx);
  test::RunDeviceQuirksTests(&ctx);
  test::RunCacheStoreTests(&ctx);
  test::RunMemoryPoolTests(&ctx);
  test::RunIntegrationTests(&ctx);

  std::cout << "[RESULT] passed=" << ctx.passed << " failed=" << ctx.failed << "\n";
  return ctx.failed == 0 ? 0 : 1;
}
