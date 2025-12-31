#include "runtime/runner.h"

#include <memory>
#include <string>

#include "lexer/lexer.h"
#include "parser/parser.h"
#include "runtime/type_checker.h"
#include "util/error.h"

namespace lattice::runtime {

ExecResult RunSource(const std::string& source, std::shared_ptr<Environment> env) {
  // Wrap in a block so multiple top-level statements are allowed.
  std::string wrapped = "{\n" + source + "\n}";
  lexer::Lexer lex(wrapped);
  parser::Parser parser(std::move(lex));
  auto stmt = parser.ParseStatement();
  TypeChecker checker;
  checker.Check(stmt.get());
  Evaluator evaluator(env);
  ExecResult result = evaluator.EvaluateStatement(*stmt);
  if (result.control == ControlSignal::kBreak || result.control == ControlSignal::kContinue) {
    throw util::Error("break/continue outside of loop", 0, 0);
  }
  return result;
}

}  // namespace lattice::runtime
