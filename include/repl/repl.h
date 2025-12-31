#ifndef LATTICE_REPL_REPL_H_
#define LATTICE_REPL_REPL_H_

#include <iostream>
#include <memory>
#include <string>

#include "builtin/builtins.h"
#include "parser/parser.h"
#include "runtime/ops.h"
#include "util/string.h"

namespace lattice::repl {

class Repl {
 public:
  /// Initializes the REPL with builtin constants/functions.
  Repl();

  /// Starts the interactive loop until EOF or "exit".
  void Run();

 private:
  /// Processes one line; returns true when the loop should terminate.
  bool ProcessLine(const std::string& line);

  std::shared_ptr<runtime::Environment> env_;
};

}  // namespace lattice::repl

#endif  // LATTICE_REPL_REPL_H_
