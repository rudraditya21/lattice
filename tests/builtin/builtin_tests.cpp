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

  auto g = EvalExpr("gamma(5)", &env);
  ExpectNear(g.f64, 24.0, "gamma_func", ctx);
  auto b = EvalExpr("beta(2,3)", &env);
  ExpectNear(b.f64, 1.0 / 12.0, "beta_func", ctx);
  auto erfv = EvalExpr("erf(0)", &env);
  ExpectNear(erfv.f64, 0.0, "erf_zero", ctx);
  auto ig = EvalExpr("igamma(2,1)", &env);
  ExpectTrue(ig.f64 > 0.0 && ig.f64 < 1.0, "igamma_range", ctx);

  auto npdf = EvalExpr("normal_pdf(0,0,1)", &env);
  ExpectNear(npdf.f64, 1.0 / std::sqrt(2 * std::acos(-1.0)), "normal_pdf_zero", ctx);
  auto ncdf = EvalExpr("normal_cdf(0,0,1)", &env);
  ExpectNear(ncdf.f64, 0.5, "normal_cdf_zero", ctx);
  auto uni = EvalExpr("uniform_pdf(0.5,0,1)", &env);
  ExpectNear(uni.f64, 1.0, "uniform_pdf_mid", ctx);
  auto expo = EvalExpr("exponential_pdf(0,1)", &env);
  ExpectNear(expo.f64, 1.0, "exponential_pdf_zero", ctx);
  auto pmf = EvalExpr("poisson_pmf(3,2)", &env);
  ExpectTrue(pmf.f64 > 0.0, "poisson_pmf_positive", ctx);
  auto binom = EvalExpr("binomial_pmf(2,4,0.5)", &env);
  ExpectNear(binom.f64, 0.375, "binomial_pmf_value", ctx);
}

}  // namespace test
