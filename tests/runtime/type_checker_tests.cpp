#include "test_util.h"

namespace test {

void RunTypeCheckerTests(TestContext* ctx) {
  // Valid annotated function and call.
  bool ok_func = true;
  try {
    TypeCheckStmt("{ func add(a: i32, b: i32) -> i32 { return a + b; } add(1, 2); }");
  } catch (const util::Error&) {
    ok_func = false;
  }
  ExpectTrue(ok_func, "typecheck_valid_function", ctx);

  // Implicit narrowing in assignment is rejected.
  bool rejected_narrow = false;
  try {
    TypeCheckStmt("x: i32 = 1.5");
  } catch (const util::Error&) {
    rejected_narrow = true;
  }
  ExpectTrue(rejected_narrow, "typecheck_reject_narrow_assignment", ctx);

  // Implicit narrowing in call is rejected.
  bool rejected_call = false;
  try {
    TypeCheckStmt("{ func f(a: i32) -> i32 { return a; } f(2.5); }");
  } catch (const util::Error&) {
    rejected_call = true;
  }
  ExpectTrue(rejected_call, "typecheck_reject_narrow_call", ctx);

  // Explicit cast permits narrowing.
  bool allows_cast = true;
  try {
    TypeCheckStmt("{ x: i32 = cast(i32, 2.5); }");
  } catch (const util::Error&) {
    allows_cast = false;
  }
  ExpectTrue(allows_cast, "typecheck_allows_cast", ctx);

  // Dynamic/unannotated code remains permissive.
  bool dynamic_ok = true;
  try {
    TypeCheckStmt("{ x = 1; x = 2.5; y = x + 3; }");
  } catch (const util::Error&) {
    dynamic_ok = false;
  }
  ExpectTrue(dynamic_ok, "typecheck_dynamic_unannotated_ok", ctx);

  // Return type enforcement.
  bool ret_mismatch = false;
  try {
    TypeCheckStmt("{ func g() -> i32 { return 1.5; } }");
  } catch (const util::Error&) {
    ret_mismatch = true;
  }
  ExpectTrue(ret_mismatch, "typecheck_return_mismatch", ctx);

  // Binding enforcement with later use.
  bool binding_enforced = true;
  try {
    TypeCheckStmt("{ x: f64 = 1.0; x = 2.0; }");
  } catch (const util::Error&) {
    binding_enforced = false;
  }
  ExpectTrue(binding_enforced, "typecheck_binding_enforced", ctx);

  // Mixed annotated/unannotated call: annotated param enforced, unannotated param dynamic.
  bool mixed_ok = true;
  try {
    TypeCheckStmt("{ func h(a: i32, b) -> i32 { return a; } h(1, 2.5); }");
  } catch (const util::Error&) {
    mixed_ok = false;
  }
  ExpectTrue(mixed_ok, "typecheck_mixed_params", ctx);

  // Decimal/rational mixing requires casts.
  bool decimal_mix_reject = false;
  try {
    TypeCheckStmt("decimal(1.0) + 1.0");
  } catch (const util::Error&) {
    decimal_mix_reject = true;
  }
  ExpectTrue(decimal_mix_reject, "typecheck_decimal_mix_reject", ctx);

  bool rational_mix_reject = false;
  try {
    TypeCheckStmt("rational(1,2) + 1");
  } catch (const util::Error&) {
    rational_mix_reject = true;
  }
  ExpectTrue(rational_mix_reject, "typecheck_rational_mix_reject", ctx);

  bool decimal_cast_ok = true;
  try {
    TypeCheckStmt("cast(f64, decimal(1.0)) + 1.0");
  } catch (const util::Error&) {
    decimal_cast_ok = false;
  }
  ExpectTrue(decimal_cast_ok, "typecheck_decimal_cast_allows_mix", ctx);
}

}  // namespace test
