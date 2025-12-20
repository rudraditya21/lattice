#include "builtin/builtins.h"

#include <cmath>

namespace lattice::builtin {

void InstallBuiltins(runtime::Environment* env) {
  if (env == nullptr) {
    return;
  }
  const double pi = std::acos(-1.0);
  const double e = std::exp(1.0);
  env->Define("pi", runtime::Value::Number(pi));
  env->Define("e", runtime::Value::Number(e));
}

}  // namespace lattice::builtin
