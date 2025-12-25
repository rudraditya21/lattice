#include "test_util.h"

#include <string>

namespace test {

void RunPromotionTests(TestContext* ctx) {
  // Promotion table expectations (API-level).
  ExpectTrue(rt::PromoteType(rt::DType::kI32, rt::DType::kF32) == rt::DType::kF32,
             "promote_int_float_to_float", ctx);
  ExpectTrue(rt::PromoteType(rt::DType::kI32, rt::DType::kC64) == rt::DType::kC64,
             "promote_int_complex_to_complex", ctx);
  ExpectTrue(rt::PromoteType(rt::DType::kU32, rt::DType::kI16) == rt::DType::kI64,
             "promote_mixed_sign_to_signed_width", ctx);
  ExpectTrue(rt::PromoteType(rt::DType::kDecimal, rt::DType::kDecimal) == rt::DType::kDecimal,
             "promote_decimal_decimal", ctx);
  ExpectTrue(rt::PromoteType(rt::DType::kRational, rt::DType::kRational) == rt::DType::kRational,
             "promote_rational_rational", ctx);

  bool threw_cross_decimal = false;
  try {
    (void)rt::PromoteType(rt::DType::kDecimal, rt::DType::kI32);
  } catch (const util::Error&) {
    threw_cross_decimal = true;
  }
  ExpectTrue(threw_cross_decimal, "decimal_cross_type_blocks", ctx);

  // Interpreter-level promotion behavior.
  rt::Environment env;
  bt::InstallBuiltins(&env);

  auto int_add = EvalExpr("1 + 2", &env);
  ExpectTrue(int_add.type == rt::DType::kI32, "int_add_stays_i32", ctx);
  ExpectNear(int_add.i64, 3.0, "int_add_value", ctx);

  auto float_mix = EvalExpr("1 + 2.5", &env);
  ExpectTrue(float_mix.type == rt::DType::kF32 || float_mix.type == rt::DType::kF64,
             "int_float_promotes_to_float", ctx);
  ExpectNear(float_mix.f64, 3.5, "int_float_value", ctx);

  auto div_float = EvalExpr("3 / 2", &env);
  ExpectTrue(div_float.type == rt::DType::kF64, "int_division_yields_float", ctx);
  ExpectNear(div_float.f64, 1.5, "int_division_value", ctx);

  // Formatting checks.
  std::string f32_str = rt::Value::F32(3.14f).ToString();
  ExpectTrue(f32_str.find("3.14") != std::string::npos, "f32_format_contains_value", ctx);
  std::string dec_str = rt::Value::Decimal(1.0L / 3.0L).ToString();
  ExpectTrue(dec_str.find("0.333") != std::string::npos, "decimal_format_precision", ctx);
}

}  // namespace test
