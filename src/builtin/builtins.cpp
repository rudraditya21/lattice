#include "builtin/builtins.h"

#include <cmath>
#include <limits>

namespace lattice::builtin {

void InstallBuiltins(runtime::Environment* env) {
  if (env == nullptr) {
    return;
  }
  const double pi = std::acos(-1.0);
  const double e = std::exp(1.0);
  const double gamma = 0.5772156649015328606;  // Euler-Mascheroni constant.
  const double inf = std::numeric_limits<double>::infinity();
  env->Define("pi", runtime::Value::Number(pi));
  env->Define("e", runtime::Value::Number(e));
  env->Define("gamma", runtime::Value::Number(gamma));
  env->Define("inf", runtime::Value::Number(inf));
}

}  // namespace lattice::builtin
