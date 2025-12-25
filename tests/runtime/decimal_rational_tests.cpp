#include "test_util.h"

namespace test {

void RunDecimalRationalTests(TestContext* ctx) {
  rt::Environment env;
  bt::InstallBuiltins(&env);

  // Decimal precision round-trip.
  EvalStmt("set_decimal_precision(4)", &env);
  auto dec = EvalExpr("decimal(1.23456)", &env);
  ExpectTrue(dec.type == rt::DType::kDecimal, "decimal_type", ctx);
  ExpectNear(dec.number, 1.2346, "decimal_rounding", ctx);

  auto prec = EvalExpr("get_decimal_precision()", &env);
  ExpectTrue(prec.type == rt::DType::kI32, "decimal_precision_type", ctx);

  // Rational normalization and arithmetic.
  auto rat = EvalExpr("rational(2, 4)", &env);
  ExpectTrue(rat.type == rt::DType::kRational, "rational_type", ctx);
  ExpectTrue(rat.rational.num == 1 && rat.rational.den == 2, "rational_normalized", ctx);

  auto rat_add = EvalExpr("rational(1, 3) + rational(1, 6)", &env);
  ExpectTrue(rat_add.rational.num == 1 && rat_add.rational.den == 2, "rational_add", ctx);
}

}  // namespace test
