#include "repl/repl.h"

#include <cctype>
#include <iostream>
#include <string>

namespace lattice::repl {

Repl::Repl() {
  builtin::InstallBuiltins(&env_);
  builtin::InstallPrint(&env_);
}

void Repl::Run() {
  std::string line;
  while (true) {
    std::cout << "lattice> " << std::flush;
    if (!std::getline(std::cin, line)) {
      std::cout << "\n";
      break;
    }
    if (ProcessLine(line)) {
      break;
    }
  }
}

bool Repl::ProcessLine(const std::string& line) {
  std::string trimmed = util::Trim(line);
  if (trimmed.empty()) {
    return false;
  }
  if (trimmed == "exit") {
    return true;
  }
  try {
    lexer::Lexer lexer(trimmed);
    parser::Parser parser(std::move(lexer));
    auto stmt = parser.ParseStatement();
    runtime::Evaluator evaluator(&env_);
    auto result = evaluator.EvaluateStatement(*stmt);
    if (result.control == runtime::ControlSignal::kBreak ||
        result.control == runtime::ControlSignal::kContinue ||
        result.control == runtime::ControlSignal::kReturn) {
      std::cout << "Error: control flow (break/continue/return) not allowed at top level\n";
      return false;
    }
    if (result.value.has_value()) {
      std::cout << result.value->ToString() << "\n";
    }
  } catch (const util::Error& err) {
    std::cout << err.formatted() << "\n";
  } catch (const std::exception& ex) {
    std::cout << "Unhandled error: " << ex.what() << "\n";
  }
  return false;
}

}  // namespace lattice::repl
