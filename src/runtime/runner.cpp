#include "runtime/runner.h"

#include <string>

#include "lexer/lexer.h"
#include "parser/parser.h"
#include "util/error.h"

namespace lattice::runtime {

ExecResult RunSource(const std::string& source, Environment* env) {
  // Wrap in a block so multiple top-level statements are allowed.
  std::string wrapped = "{\n" + source + "\n}";
  lexer::Lexer lex(wrapped);
  parser::Parser parser(std::move(lex));
  auto stmt = parser.ParseStatement();
  Evaluator evaluator(env);
  ExecResult result = evaluator.EvaluateStatement(*stmt);
  if (result.control == ControlSignal::kBreak || result.control == ControlSignal::kContinue) {
    throw util::Error("break/continue outside of loop", 0, 0);
  }
  return result;
}

}  // namespace lattice::runtime
