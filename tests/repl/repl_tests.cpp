#include "test_util.h"

#include <istream>
#include <ostream>
#include <sstream>

#include "repl/repl.h"

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
  bool has_sum = out.find("2.000000") != std::string::npos;
  ExpectTrue(has_sum, "repl_evaluates_expression", ctx);

  std::istringstream input2("{ a = 4; if (a) a = a + 1; a }\nexit\n");
  std::ostringstream output2;
  StreamRedirect redirect2(std::cin, std::cout, input2, output2);

  lattice::repl::Repl repl2;
  repl2.Run();

  redirect2.Restore(std::cin, std::cout);
  std::string out2 = output2.str();
  bool has_block = out2.find("5.000000") != std::string::npos;
  ExpectTrue(has_block, "repl_handles_block_and_if", ctx);
}

}  // namespace test
