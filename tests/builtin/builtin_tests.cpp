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

  auto p1 = EvalExpr("philox(1234, 1, 0)", &env);
  auto p2 = EvalExpr("philox(1234, 1, 0)", &env);
  auto p3 = EvalExpr("philox(1234, 1, 1)", &env);
  ExpectNear(p1.f64, p2.f64, "philox_deterministic", ctx);
  ExpectTrue(std::abs(p1.f64 - p3.f64) > 1e-9, "philox_counter_changes", ctx);

  auto t1 = EvalExpr("threefry(42, 7, 0)", &env);
  auto t2 = EvalExpr("threefry(42, 7, 1)", &env);
  ExpectTrue(std::abs(t1.f64 - t2.f64) > 1e-9, "threefry_counter_changes", ctx);
  auto t3 = EvalExpr("threefry(42, 7, 0)", &env);
  ExpectNear(t1.f64, t3.f64, "threefry_deterministic", ctx);
}

}  // namespace test
