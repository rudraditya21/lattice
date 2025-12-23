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
}

}  // namespace test
