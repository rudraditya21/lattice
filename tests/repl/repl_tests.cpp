#include <istream>
#include <ostream>
#include <sstream>

#include "repl/repl.h"
#include "test_util.h"

namespace test {

class StreamRedirect {
 public:
  StreamRedirect(std::istream& in, std::ostream& out, std::istream& new_in, std::ostream& new_out)
      : orig_in_buf_(in.rdbuf()), orig_out_buf_(out.rdbuf()) {
    in.rdbuf(new_in.rdbuf());
    out.rdbuf(new_out.rdbuf());
  }

  ~StreamRedirect() = default;

  void Restore(std::istream& in, std::ostream& out) {
    in.rdbuf(orig_in_buf_);
    out.rdbuf(orig_out_buf_);
  }

 private:
  std::streambuf* orig_in_buf_;
  std::streambuf* orig_out_buf_;
};

void RunReplTests(TestContext* ctx) {
  std::istringstream input("1 + 1\nexit\n");
  std::ostringstream output;
  StreamRedirect redirect(std::cin, std::cout, input, output);

  lattice::repl::Repl repl;
  repl.Run();

  redirect.Restore(std::cin, std::cout);

  std::string out = output.str();
  bool has_sum = out.find("2") != std::string::npos;
  ExpectTrue(has_sum, "repl_evaluates_expression", ctx);

  std::istringstream input2("{ a = 4; if (a) a = a + 1; a }\nexit\n");
  std::ostringstream output2;
  StreamRedirect redirect2(std::cin, std::cout, input2, output2);

  lattice::repl::Repl repl2;
  repl2.Run();

  redirect2.Restore(std::cin, std::cout);
  std::string out2 = output2.str();
  bool has_block = out2.find("5") != std::string::npos;
  ExpectTrue(has_block, "repl_handles_block_and_if", ctx);

  std::istringstream input3("{ n = 0; while (n - 2) { n = n + 1; }; n }\nexit\n");
  std::ostringstream output3;
  StreamRedirect redirect3(std::cin, std::cout, input3, output3);
  lattice::repl::Repl repl3;
  repl3.Run();
  redirect3.Restore(std::cin, std::cout);
  std::string out3 = output3.str();
  bool has_loop = out3.find("2") != std::string::npos;
  ExpectTrue(has_loop, "repl_handles_while", ctx);

  std::istringstream input4(
      "{ s = 0; for (i = 0; i - 3; i = i + 1) { if (i - 1) { } else { continue; } s = s + 1; }; s "
      "}\nexit\n");
  std::ostringstream output4;
  StreamRedirect redirect4(std::cin, std::cout, input4, output4);
  lattice::repl::Repl repl4;
  repl4.Run();
  redirect4.Restore(std::cin, std::cout);
  std::string out4 = output4.str();
  bool has_for = out4.find("2") != std::string::npos;
  ExpectTrue(has_for, "repl_handles_for_and_continue", ctx);
}

}  // namespace test
