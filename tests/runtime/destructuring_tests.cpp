#include "test_util.h"

namespace test {

void RunDestructuringTests(TestContext* ctx) {
  rt::Environment env;
  bt::InstallBuiltins(&env);

  // Tuple destructuring.
  EvalStmt("(a, b) = (1, 2)", &env);
  auto a = EvalExpr("a", &env);
  auto b = EvalExpr("b", &env);
  ExpectNear(static_cast<double>(a.i64), 1.0, "destructure_tuple_a", ctx);
  ExpectNear(static_cast<double>(b.i64), 2.0, "destructure_tuple_b", ctx);

  bool tuple_err = false;
  try {
    EvalStmt("(x, y) = {k: 1}", &env);
  } catch (const util::Error&) {
    tuple_err = true;
  }
  ExpectTrue(tuple_err, "destructure_tuple_wrong_rhs", ctx);
  bool tuple_arity_err = false;
  try {
    EvalStmt("(m, n, o) = (1, 2)", &env);
  } catch (const util::Error&) {
    tuple_arity_err = true;
  }
  ExpectTrue(tuple_arity_err, "destructure_tuple_arity_mismatch", ctx);

  // Record destructuring.
  EvalStmt("{x, y} = {x: 3, y: 4}", &env);
  auto x = EvalExpr("x", &env);
  auto y = EvalExpr("y", &env);
  ExpectNear(static_cast<double>(x.i64), 3.0, "destructure_record_x", ctx);
  ExpectNear(static_cast<double>(y.i64), 4.0, "destructure_record_y", ctx);

  bool record_err = false;
  try {
    EvalStmt("{p, q} = {p: 1}", &env);
  } catch (const util::Error&) {
    record_err = true;
  }
  ExpectTrue(record_err, "destructure_record_missing", ctx);

  bool record_type_err = false;
  try {
    EvalStmt("{a} = (1,)", &env);
  } catch (const util::Error&) {
    record_type_err = true;
  }
  ExpectTrue(record_type_err, "destructure_record_wrong_type", ctx);
}

}  // namespace test
