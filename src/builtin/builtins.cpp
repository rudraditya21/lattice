#include "builtin/builtins.h"

#include <cmath>
#include <limits>

namespace lattice::builtin {

void InstallBuiltins(const std::shared_ptr<runtime::Environment>& env) {
  if (!env) {
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

void InstallPrint(const std::shared_ptr<runtime::Environment>& env) {
  if (!env) {
    return;
  }
  auto print_fn = std::make_shared<runtime::Function>();
  print_fn->parameters = {"x"};
  print_fn->parameter_types = {""};
  print_fn->return_type = "";
  print_fn->body = nullptr;  // Special-cased in evaluator.
  print_fn->defining_env = env;
  env->Define("print", runtime::Value::Func(print_fn));
}

}  // namespace lattice::builtin
