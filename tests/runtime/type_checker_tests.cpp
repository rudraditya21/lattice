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
}

}  // namespace test
