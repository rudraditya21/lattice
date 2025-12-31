#include "test_util.h"

namespace test {

void RunComplexTests(TestContext* ctx) {
  auto env = std::make_shared<rt::Environment>();
  bt::InstallBuiltins(env);

  auto csum = EvalExpr("complex(1, 2) + complex(3, 4)", env);
  ExpectTrue(csum.type == rt::DType::kC128, "complex_sum_type", ctx);
  ExpectTrue(csum.complex.real() == 4 && csum.complex.imag() == 6, "complex_sum_value", ctx);

  auto cmul = EvalExpr("complex(1, 1) * complex(1, -1)", env);
  ExpectTrue(cmul.complex.real() == 2 && cmul.complex.imag() == 0, "complex_mul_value", ctx);

  auto mixed = EvalExpr("complex(1, 0) + 2.0", env);
  ExpectTrue(mixed.type == rt::DType::kC128, "complex_float_promotes", ctx);

  auto to_str = EvalExpr("complex(1, -2)", env).ToString();
  ExpectTrue(to_str.find("1") != std::string::npos && to_str.find("-2") != std::string::npos,
             "complex_to_string", ctx);
}

}  // namespace test
