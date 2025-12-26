#include "test_util.h"

namespace test {

void RunTupleRecordTests(TestContext* ctx) {
  rt::Environment env;
  bt::InstallBuiltins(&env);

  // Tuple literal and indexing.
  auto tup = EvalExpr("(1, 2.5, \"x\")", &env);
  ExpectTrue(tup.type == rt::DType::kTuple, "tuple_type", ctx);
  ExpectTrue(tup.tuple.elements.size() == 3, "tuple_len", ctx);
  auto first = EvalExpr("(1, 2)[0]", &env);
  ExpectTrue(first.type == rt::DType::kI32, "tuple_index_type", ctx);
  ExpectNear(first.i64, 1.0, "tuple_index_value", ctx);

  bool bounds_err = false;
  try {
    (void)EvalExpr("(1,)[1]", &env);
  } catch (const util::Error&) {
    bounds_err = true;
  }
  ExpectTrue(bounds_err, "tuple_index_bounds", ctx);

  // Record literal and access.
  auto rec = EvalExpr("{x: 1, \"y\": 3.5}", &env);
  ExpectTrue(rec.type == rt::DType::kRecord, "record_type", ctx);
  ExpectTrue(rec.record.fields.size() == 2, "record_len", ctx);
  auto rec_x = EvalExpr("{x: 1, \"y\": 3.5}[\"x\"]", &env);
  ExpectNear(rec_x.i64, 1.0, "record_access_value", ctx);

  bool key_err = false;
  try {
    (void)EvalExpr("{x: 1}[\"missing\"]", &env);
  } catch (const util::Error&) {
    key_err = true;
  }
  ExpectTrue(key_err, "record_missing_key", ctx);

  // Structural equality.
  auto tup_eq = EvalExpr("(1, 2) == (1, 2)", &env);
  ExpectTrue(tup_eq.boolean, "tuple_equality_true", ctx);
  auto rec_eq = EvalExpr("{a: 1} == {a: 2}", &env);
  ExpectTrue(!rec_eq.boolean, "record_equality_false", ctx);

  // Builtins.
  auto l1 = EvalExpr("len((1,2,3))", &env);
  ExpectNear(l1.i64, 3.0, "len_tuple", ctx);
  auto k1 = EvalExpr("keys({x:1, y:2})", &env);
  ExpectTrue(k1.type == rt::DType::kTuple, "keys_tuple_type", ctx);
  auto v1 = EvalExpr("values({x:1, y:2})", &env);
  ExpectTrue(v1.type == rt::DType::kTuple, "values_tuple_type", ctx);
  auto hk = EvalExpr("has_key({x:1}, \"x\")", &env);
  ExpectTrue(hk.boolean, "has_key_true", ctx);
}

}  // namespace test
