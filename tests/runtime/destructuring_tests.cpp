#include "test_util.h"

namespace test {

void RunDestructuringTests(TestContext* ctx) {
  rt::Environment env;
  bt::InstallBuiltins(&env);

  // Tuple destructuring.
  EvalStmt("(a, b) = (1, 2)", &env);
  auto a = EvalExpr("a", &env);
  auto b = EvalExpr("b", &env);
  ExpectNear(a.i64, 1.0, "destructure_tuple_a", ctx);
  ExpectNear(b.i64, 2.0, "destructure_tuple_b", ctx);

  bool tuple_err = false;
  try {
    EvalStmt("(x, y) = {k: 1}", &env);
  } catch (const util::Error&) {
    tuple_err = true;
  }
  ExpectTrue(tuple_err, "destructure_tuple_wrong_rhs", ctx);

  // Record destructuring.
  EvalStmt("{x, y} = {x: 3, y: 4}", &env);
  auto x = EvalExpr("x", &env);
  auto y = EvalExpr("y", &env);
  ExpectNear(x.i64, 3.0, "destructure_record_x", ctx);
  ExpectNear(y.i64, 4.0, "destructure_record_y", ctx);

  bool record_err = false;
  try {
    EvalStmt("{p, q} = {p: 1}", &env);
  } catch (const util::Error&) {
    record_err = true;
  }
  ExpectTrue(record_err, "destructure_record_missing", ctx);
}

}  // namespace test
