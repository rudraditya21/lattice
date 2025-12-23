#ifndef LATTICE_BUILTIN_BUILTINS_H_
#define LATTICE_BUILTIN_BUILTINS_H_

#include "runtime/environment.h"

namespace lattice::builtin {

/// Registers builtin constants (pi, e, gamma, inf) and math functions into the provided
/// environment.
void InstallBuiltins(runtime::Environment* env);

}  // namespace lattice::builtin

#endif  // LATTICE_BUILTIN_BUILTINS_H_
