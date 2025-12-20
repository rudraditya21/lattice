#include "repl/repl.h"

#include <cctype>
#include <iostream>
#include <string>
#include <unordered_set>

namespace lattice::repl {

Repl::Repl() {
  builtin::InstallBuiltins(&env_);
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
  auto is_identifier = [](const std::string& name) -> bool {
    if (name.empty()) {
      return false;
    }
    if (!std::isalpha(static_cast<unsigned char>(name[0])) && name[0] != '_') {
      return false;
    }
    for (char ch : name) {
      if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_') {
        return false;
      }
    }
    return true;
  };
  static const std::unordered_set<std::string> kReserved = {
      "pi",  "e",     "pow",  "gcd",   "lcm",   "abs", "sign",
      "mod", "floor", "ceil", "round", "clamp", "min", "max"};
  try {
    size_t eq_pos = trimmed.find('=');
    if (eq_pos != std::string::npos) {
      std::string lhs = util::Trim(trimmed.substr(0, eq_pos));
      std::string rhs = util::Trim(trimmed.substr(eq_pos + 1));
      if (!is_identifier(lhs)) {
        std::cout << "Error: invalid identifier on left-hand side\n";
        return false;
      }
      if (kReserved.find(lhs) != kReserved.end()) {
        std::cout << "Error: '" << lhs << "' is reserved\n";
        return false;
      }
      if (rhs.empty()) {
        std::cout << "Error: right-hand side of assignment is empty\n";
        return false;
      }
      lexer::Lexer rhs_lexer(rhs);
      parser::Parser rhs_parser(std::move(rhs_lexer));
      auto rhs_expr = rhs_parser.ParseExpression();
      runtime::Evaluator evaluator(&env_);
      runtime::Value value = evaluator.Evaluate(*rhs_expr);
      env_.Define(lhs, value);
      std::cout << value.ToString() << "\n";
      return false;
    }

    lexer::Lexer lexer(trimmed);
    parser::Parser parser(std::move(lexer));
    auto expr = parser.ParseExpression();
    runtime::Evaluator evaluator(&env_);
    runtime::Value value = evaluator.Evaluate(*expr);
    std::cout << value.ToString() << "\n";
  } catch (const util::Error& err) {
    std::cout << err.formatted() << "\n";
  } catch (const std::exception& ex) {
    std::cout << "Unhandled error: " << ex.what() << "\n";
  }
  return false;
}

}  // namespace lattice::repl
