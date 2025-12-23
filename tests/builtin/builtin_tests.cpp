#include "test_util.h"

namespace test {

void RunBuiltinTests(TestContext* ctx) {
  rt::Environment env;
  bt::InstallBuiltins(&env);
  auto pi = EvalExpr("pi", &env);
  auto e = EvalExpr("e", &env);
  auto gamma = EvalExpr("gamma", &env);
  auto inf = EvalExpr("inf", &env);
  ExpectNear(pi.number, std::acos(-1.0), "pi_constant", ctx);
  ExpectNear(e.number, std::exp(1.0), "e_constant", ctx);
  ExpectNear(gamma.number, 0.5772156649015328606, "gamma_constant", ctx);
  ExpectTrue(std::isinf(inf.number), "inf_constant", ctx);
}

}  // namespace test
