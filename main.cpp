// Entry point for the lattice REPL.
#include "repl/repl.h"

int main() {
  lattice::repl::Repl repl;
  repl.Run();
  return 0;
}
