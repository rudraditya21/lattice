#include "test_util.h"
#include "util/string.h"

namespace test {

void RunUtilTests(TestContext* ctx) {
  std::string trimmed = lattice::util::Trim("  hello \n");
  ExpectTrue(trimmed == "hello", "trim_basic", ctx);

  std::vector<std::string> lines = lattice::util::SplitLines("a\nb\n");
  ExpectTrue(lines.size() == 3, "split_lines_len", ctx);
  ExpectTrue(lines[0] == "a" && lines[1] == "b" && lines[2].empty(), "split_lines_values", ctx);

  lattice::util::Error err("boom", 2, 4);
  std::string formatted = err.formatted();
  ExpectTrue(formatted.find("2:4") != std::string::npos, "error_formatted_location", ctx);
  ExpectTrue(formatted.find("boom") != std::string::npos, "error_formatted_message", ctx);
}

}  // namespace test
