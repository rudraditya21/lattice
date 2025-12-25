#include "test_util.h"
#include "util/string.h"

namespace test {

void RunUtilTests(TestContext* ctx) {
  std::string trimmed = lattice::util::Trim("  hello \n");
  ExpectTrue(trimmed == "hello", "trim_basic", ctx);
}

}  // namespace test
