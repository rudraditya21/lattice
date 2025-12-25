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
  ExpectTrue(rat_add.ToString() == "1/2", "rational_to_string", ctx);

  // Mixing with floats/complex should be explicit only.
  bool decimal_mix_error = false;
  try {
    EvalExpr("decimal(1.0) + 1.0", &env);
  } catch (const util::Error&) {
    decimal_mix_error = true;
  }
  ExpectTrue(decimal_mix_error, "decimal_float_mix_error", ctx);

  bool rational_mix_error = false;
  try {
    EvalExpr("rational(1,2) + 1", &env);
  } catch (const util::Error&) {
    rational_mix_error = true;
  }
  ExpectTrue(rational_mix_error, "rational_int_mix_error", ctx);

  // Explicit conversion works.
  bool cast_ok = true;
  try {
    EvalExpr("cast(f64, decimal(1.25)) + 1.0", &env);
  } catch (const util::Error&) {
    cast_ok = false;
  }
  ExpectTrue(cast_ok, "explicit_cast_allows_mix", ctx);

  // ToString respects precision.
  EvalStmt("set_decimal_precision(3)", &env);
  auto dec_str = EvalExpr("decimal(1.23456)", &env).ToString();
  ExpectTrue(dec_str.find("1.235") != std::string::npos, "decimal_precision_string", ctx);
}

}  // namespace test
