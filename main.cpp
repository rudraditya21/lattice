// Entry point for the lattice REPL or script runner.
#include <fstream>
#include <iostream>
#include <sstream>

#include "builtin/builtins.h"
#include "repl/repl.h"
#include "runtime/runner.h"
#include "util/error.h"

int main(int argc, char** argv) {
  // Simple CLI: no args -> REPL; arg[1] -> run script file.
  if (argc > 1) {
    std::ifstream in(argv[1]);
    if (!in) {
      std::cerr << "Could not open file: " << argv[1] << "\n";
      return 1;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    lattice::runtime::Environment env;
    lattice::builtin::InstallBuiltins(&env);
    lattice::builtin::InstallPrint(&env);
    try {
      lattice::runtime::ExecResult result =
          lattice::runtime::RunSource(buffer.str(), &env);
      (void)result;  // Script mode only prints via explicit print().
      return 0;
    } catch (const lattice::util::Error& err) {
      std::cerr << err.formatted() << "\n";
      return 1;
    } catch (const std::exception& ex) {
      std::cerr << "Unhandled error: " << ex.what() << "\n";
      return 1;
    }
  } else {
    lattice::repl::Repl repl;
    repl.Run();
    return 0;
  }
}
