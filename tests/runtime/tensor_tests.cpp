#include "test_util.h"

namespace test {

void RunTensorTests(TestContext* ctx) {
  rt::Environment env;
  bt::InstallBuiltins(&env);

  auto t1 = EvalExpr("tensor(3, 1)", &env);  // shape (3,) fill 1
  ExpectTrue(t1.type == rt::DType::kTensor, "tensor_type", ctx);
  ExpectTrue(t1.tensor.shape.size() == 1 && t1.tensor.shape[0] == 3, "tensor_shape_1d", ctx);
  ExpectTrue(t1.tensor.elem_type == rt::DType::kI32 || t1.tensor.elem_type == rt::DType::kF64,
             "tensor_elem_type", ctx);
  ExpectTrue(t1.tensor.size == 3, "tensor_storage_size", ctx);

  auto t2 = EvalExpr("tensor(2, 3, 0)", &env);  // shape (2,3) fill 0
  ExpectTrue(t2.tensor.shape.size() == 2 && t2.tensor.shape[0] == 2 && t2.tensor.shape[1] == 3,
             "tensor_shape_2d", ctx);
  // Row-major strides: [3,1]
  ExpectTrue(
      t2.tensor.strides.size() == 2 && t2.tensor.strides[0] == 3 && t2.tensor.strides[1] == 1,
      "tensor_strides_row_major", ctx);
  ExpectTrue(t2.tensor.size == 6, "tensor_storage_size_2d", ctx);
  auto desc = t2.ToString();
  ExpectTrue(desc.find("tensor") != std::string::npos, "tensor_to_string", ctx);

  // Elementwise add with scalar broadcast and sum/mean.
  auto t3_stmt = EvalStmt("{ t3 = tensor(2, 2, 1); t3; }", &env);
  auto t3 = Unwrap(t3_stmt.value, "tensor_values_stmt", ctx);
  auto t4 = EvalExpr("t3 + 1", &env);
  ExpectTrue(t4.tensor.size == 4, "tensor_broadcast_size", ctx);
  double first_val = t4.tensor.using_inline ? t4.tensor.inline_storage[0] : t4.tensor.storage[0];
  ExpectNear(first_val, 2.0, "tensor_broadcast_add", ctx);
  auto tsum = EvalExpr("sum(t3)", &env);
  ExpectNear(tsum.f64, 4.0, "tensor_sum", ctx);
  auto tmean = EvalExpr("mean(t3)", &env);
  ExpectNear(tmean.f64, 1.0, "tensor_mean", ctx);

  // Dtype metadata.
  ExpectTrue(t3.tensor.elem_type == rt::DType::kF64, "tensor_elem_dtype", ctx);

  // Literal construction and dtype promotion.
  auto tvals = EvalExpr("tensor_values(1, 2, 3)", &env);
  ExpectTrue(tvals.tensor.shape[0] == 3, "tensor_values_shape", ctx);
  ExpectTrue(tvals.tensor.elem_type == rt::DType::kI32 || tvals.tensor.elem_type == rt::DType::kI64,
             "tensor_values_elem_type", ctx);
  auto tvals_mix = EvalExpr("tensor_values(1, 2.5)", &env);
  ExpectTrue(tvals_mix.tensor.elem_type == rt::DType::kF32 ||
                 tvals_mix.tensor.elem_type == rt::DType::kF64,
             "tensor_values_promotion", ctx);

  // Sparse CSR construction and to_dense.
  auto csr = EvalExpr("tensor_sparse_csr((2,2), (0,1,2), (0,1), (1,2))", &env);
  ExpectTrue(csr.tensor.kind == rt::TensorKind::kSparseCSR, "csr_kind", ctx);
  auto csr_dense = EvalExpr("to_dense(tensor_sparse_csr((2,2), (0,1,2), (0,1), (1,2)))", &env);
  double* cd = csr_dense.tensor.Data();
  ExpectNear(cd[0], 1.0, "csr_dense_value0", ctx);
  ExpectNear(cd[3], 2.0, "csr_dense_value_last", ctx);

  // Sparse COO construction and to_dense.
  auto coo_dense = EvalExpr("to_dense(tensor_sparse_coo((2,2), (0,1), (0,1), (3,4)))", &env);
  double* cod = coo_dense.tensor.Data();
  ExpectNear(cod[0], 3.0, "coo_dense_value0", ctx);
  ExpectNear(cod[3], 4.0, "coo_dense_value_last", ctx);

  // to_sparse conversions.
  auto to_csr = EvalExpr("to_sparse_csr(tensor(2,2,1))", &env);
  ExpectTrue(to_csr.tensor.kind == rt::TensorKind::kSparseCSR, "to_csr_kind", ctx);
  auto to_coo = EvalExpr("to_sparse_coo(tensor(2,2,1))", &env);
  ExpectTrue(to_coo.tensor.kind == rt::TensorKind::kSparseCOO, "to_coo_kind", ctx);

  // Dense + sparse (densify sparse) works.
  auto dense_sparse = EvalExpr("tensor(2,2,1) + to_sparse_csr(tensor(2,2,2))", &env);
  ExpectTrue(dense_sparse.tensor.kind == rt::TensorKind::kDense, "dense_sparse_dense_out", ctx);
  ExpectNear(dense_sparse.tensor.Data()[0], 3.0, "dense_sparse_value", ctx);

  // Dense + sparse with broadcast (densify sparse then broadcast dense).
  auto dense_sparse_bcast = EvalExpr("tensor(1,1,5) + to_sparse_csr(tensor(2,2,1))", &env);
  ExpectTrue(dense_sparse_bcast.tensor.shape[0] == 2 && dense_sparse_bcast.tensor.shape[1] == 2,
             "dense_sparse_broadcast_shape", ctx);
  ExpectNear(dense_sparse_bcast.tensor.Data()[0], 6.0, "dense_sparse_broadcast_value", ctx);

  // Sparse + sparse (same format) elementwise.
  auto csr_add = EvalExpr(
      "tensor_sparse_csr((2,2), (0,1,2), (0,1), (1,2)) + tensor_sparse_csr((2,2), (0,1,2), (0,1), "
      "(3,4))",
      &env);
  ExpectTrue(csr_add.tensor.kind == rt::TensorKind::kSparseCSR, "csr_add_kind", ctx);
  auto csr_add_dense = EvalExpr(
      "to_dense(tensor_sparse_csr((2,2), (0,1,2), (0,1), (1,2)) + "
      "tensor_sparse_csr((2,2), (0,1,2), (0,1), (3,4)))",
      &env);
  ExpectNear(csr_add_dense.tensor.Data()[0], 4.0, "csr_add_value0", ctx);
  ExpectNear(csr_add_dense.tensor.Data()[3], 6.0, "csr_add_value_last", ctx);

  auto coo_add = EvalExpr(
      "tensor_sparse_coo((2,2), (0,1), (0,1), (1,2)) + tensor_sparse_coo((2,2), (0,1), (0,1), "
      "(3,4))",
      &env);
  ExpectTrue(coo_add.tensor.kind == rt::TensorKind::kSparseCOO, "coo_add_kind", ctx);

  // Sparse shape mismatch errors.
  bool sparse_shape_err = false;
  try {
    EvalExpr(
        "tensor_sparse_csr((2,2), (0,1,2), (0,1), (1,2)) + tensor_sparse_csr((3,3), (0,1,2,2), "
        "(0,1), (1,2))",
        &env);
  } catch (const util::Error&) {
    sparse_shape_err = true;
  }
  ExpectTrue(sparse_shape_err, "sparse_shape_mismatch_error", ctx);

  // Sparse format mismatch errors.
  bool sparse_format_err = false;
  try {
    EvalExpr(
        "tensor_sparse_csr((2,2), (0,1,1), (0), (1)) + "
        "tensor_sparse_coo((2,2), (0), (0), (1))",
        &env);
  } catch (const util::Error&) {
    sparse_format_err = true;
  }
  ExpectTrue(sparse_format_err, "sparse_format_mismatch_error", ctx);

  // Ragged elementwise (matching splits) and error cases.
  auto rag_ok =
      EvalExpr("tensor_ragged((0,2,4), (1,2,3,4)) + tensor_ragged((0,2,4), (1,1,1,1))", &env);
  ExpectTrue(rag_ok.tensor.kind == rt::TensorKind::kRagged, "ragged_kind", ctx);
  ExpectNear(rag_ok.tensor.ragged_values[0], 2.0, "ragged_add_value0", ctx);
  bool rag_mismatch = false;
  try {
    EvalExpr("tensor_ragged((0,2), (1,2)) + tensor_ragged((0,3), (1,2,3))", &env);
  } catch (const util::Error&) {
    rag_mismatch = true;
  }
  ExpectTrue(rag_mismatch, "ragged_mismatch_error", ctx);
  bool rag_dense_mix = false;
  try {
    EvalExpr("tensor_ragged((0,2), (1,2)) + tensor(2,1)", &env);
  } catch (const util::Error&) {
    rag_dense_mix = true;
  }
  ExpectTrue(rag_dense_mix, "ragged_dense_mix_error", ctx);

  // Broadcast across higher ranks.
  auto tb = EvalExpr("tensor(2, 1, 1) + tensor(1, 3, 2)", &env);  // (2,1) + (1,3)
  ExpectTrue(tb.tensor.shape.size() == 2 && tb.tensor.shape[0] == 2 && tb.tensor.shape[1] == 3,
             "tensor_broadcast_shape", ctx);
  double* tb_data = tb.tensor.Data();
  ExpectNear(tb_data[0], 3.0, "tensor_broadcast_value_0", ctx);
  ExpectNear(tb_data[tb.tensor.size - 1], 3.0, "tensor_broadcast_value_last", ctx);

  // Error on incompatible shapes for elementwise tensor-tensor op.
  bool shape_error = false;
  try {
    EvalExpr("tensor(2,2,1) + tensor(3,1)", &env);
  } catch (const util::Error&) {
    shape_error = true;
  }
  ExpectTrue(shape_error, "tensor_shape_mismatch_error", ctx);

  bool broadcast_error = false;
  try {
    EvalExpr("tensor(2, 3, 1) + tensor(4, 1)", &env);
  } catch (const util::Error&) {
    broadcast_error = true;
  }
  ExpectTrue(broadcast_error, "tensor_broadcast_incompatible_error", ctx);

  bool tensor_div_zero = false;
  try {
    EvalExpr("tensor(2, 1, 1) / tensor(2, 1, 0)", &env);
  } catch (const util::Error&) {
    tensor_div_zero = true;
  }
  ExpectTrue(tensor_div_zero, "tensor_division_by_zero_errors", ctx);
}

}  // namespace test
