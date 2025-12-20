#ifndef LATTICE_BUILTIN_BUILTINS_H_
#define LATTICE_BUILTIN_BUILTINS_H_

#include "runtime/environment.h"

namespace lattice::builtin {

void InstallBuiltins(runtime::Environment* env);

}  // namespace lattice::builtin

#endif  // LATTICE_BUILTIN_BUILTINS_H_
