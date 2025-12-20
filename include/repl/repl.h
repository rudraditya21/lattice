#ifndef LATTICE_REPL_REPL_H_
#define LATTICE_REPL_REPL_H_

#include <iostream>
#include <string>

#include "builtin/builtins.h"
#include "parser/parser.h"
#include "runtime/ops.h"
#include "util/string.h"

namespace lattice::repl {

class Repl {
 public:
  Repl();
  void Run();

 private:
  bool ProcessLine(const std::string& line);

  runtime::Environment env_;
};

}  // namespace lattice::repl

#endif  // LATTICE_REPL_REPL_H_
