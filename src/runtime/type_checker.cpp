#include "runtime/type_checker.h"

#include <numeric>

#include "runtime/ops.h"
#include "util/error.h"

namespace lattice::runtime {

namespace {
std::optional<DType> LookupDType(const std::string& name) {
  if (name == "bool") return DType::kBool;
  if (name == "i8") return DType::kI8;
  if (name == "i16") return DType::kI16;
  if (name == "i32") return DType::kI32;
  if (name == "i64") return DType::kI64;
  if (name == "u8") return DType::kU8;
  if (name == "u16") return DType::kU16;
  if (name == "u32") return DType::kU32;
  if (name == "u64") return DType::kU64;
  if (name == "f16") return DType::kF16;
  if (name == "bfloat16") return DType::kBF16;
  if (name == "f32") return DType::kF32;
  if (name == "f64") return DType::kF64;
  if (name == "complex64") return DType::kC64;
  if (name == "complex128") return DType::kC128;
  if (name == "decimal") return DType::kDecimal;
  if (name == "rational") return DType::kRational;
  return std::nullopt;
}
}  // namespace

TypeChecker::TypeChecker() {
  EnterScope();
  // Minimal builtin signatures to allow checking common calls.
  functions_["print"] = FunSig{{std::nullopt}, std::nullopt};
  functions_["pow"] = FunSig{{std::nullopt, std::nullopt}, DType::kF64};
  functions_["gcd"] = FunSig{{std::nullopt, std::nullopt}, DType::kI64};
  functions_["lcm"] = FunSig{{std::nullopt, std::nullopt}, DType::kI64};
  functions_["abs"] = FunSig{{std::nullopt}, std::nullopt};
  functions_["set_decimal_precision"] = FunSig{{DType::kI32}, std::nullopt};
  functions_["get_decimal_precision"] = FunSig{{}, DType::kI32};
  functions_["int"] = FunSig{{std::nullopt}, DType::kI64};
  functions_["float"] = FunSig{{std::nullopt}, DType::kF64};
  functions_["decimal"] = FunSig{{std::nullopt}, DType::kDecimal};
  functions_["rational"] = FunSig{{std::nullopt, std::nullopt}, DType::kRational};
  functions_["complex"] = FunSig{{std::nullopt, std::nullopt}, DType::kC128};
  functions_["tensor"] = FunSig{{std::nullopt, std::nullopt}, DType::kTensor};
  functions_["sum"] = FunSig{{DType::kTensor}, DType::kF64};
  functions_["mean"] = FunSig{{DType::kTensor}, DType::kF64};
}

void TypeChecker::EnterScope() {
  scopes_.emplace_back();
}

void TypeChecker::ExitScope() {
  scopes_.pop_back();
}

bool TypeChecker::IsAssignable(std::optional<DType> from, std::optional<DType> to) {
  if (!to.has_value()) return true;
  if (!from.has_value()) return true;
  if (to.value() == DType::kTensor) {
    return from.value() == DType::kTensor;
  }
  if (from.value() == to.value()) return true;
  return PromoteType(from.value(), to.value()) == to.value();
}

std::optional<DType> TypeChecker::TypeOf(const parser::Expression* expr) {
  using parser::BinaryExpression;
  using parser::BinaryOp;
  using parser::BoolLiteral;
  using parser::CallExpression;
  using parser::Identifier;
  using parser::NumberLiteral;
  using parser::UnaryExpression;
  using parser::UnaryOp;

  if (auto num = dynamic_cast<const NumberLiteral*>(expr)) {
    if (num->is_integer_token) return DType::kI32;
    size_t digits = 0;
    for (char c : num->lexeme) {
      if (std::isdigit(static_cast<unsigned char>(c))) ++digits;
    }
    return digits <= 7 ? DType::kF32 : DType::kF64;
  }
  if (dynamic_cast<const BoolLiteral*>(expr)) {
    return DType::kBool;
  }
  if (auto id = dynamic_cast<const Identifier*>(expr)) {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      auto found = it->find(id->name);
      if (found != it->end()) {
        return found->second;
      }
    }
    return std::nullopt;
  }
  if (auto unary = dynamic_cast<const UnaryExpression*>(expr)) {
    return TypeOf(unary->operand.get());
  }
  if (auto bin = dynamic_cast<const BinaryExpression*>(expr)) {
    auto lt = TypeOf(bin->lhs.get());
    auto rt = TypeOf(bin->rhs.get());
    if (!lt.has_value() || !rt.has_value()) return std::nullopt;
    if (bin->op == BinaryOp::kEq || bin->op == BinaryOp::kNe || bin->op == BinaryOp::kGt ||
        bin->op == BinaryOp::kGe || bin->op == BinaryOp::kLt || bin->op == BinaryOp::kLe) {
      return DType::kBool;
    }
    return PromoteType(lt.value(), rt.value());
  }
  if (auto call = dynamic_cast<const CallExpression*>(expr)) {
    if (call->callee == "cast") {
      if (call->args.size() != 2) {
        throw util::Error("cast expects two arguments: type name and expression", 0, 0);
      }
      auto* type_id = dynamic_cast<Identifier*>(call->args[0].get());
      if (type_id == nullptr) {
        throw util::Error("cast first argument must be a type name identifier", 0, 0);
      }
      auto dt = LookupDType(type_id->name);
      if (!dt.has_value()) {
        throw util::Error("Unknown cast target type: " + type_id->name, 0, 0);
      }
      return dt;
    }
    auto fn = functions_.find(call->callee);
    if (fn != functions_.end()) {
      const auto& sig = fn->second;
      if (sig.params.size() == call->args.size()) {
        for (size_t i = 0; i < call->args.size(); ++i) {
          auto arg_t = TypeOf(call->args[i].get());
          if (!IsAssignable(arg_t, sig.params[i])) {
            throw util::Error(
                "Type mismatch for argument " + std::to_string(i + 1) + " to " + call->callee, 0,
                0);
          }
        }
      }
      return sig.ret;
    }
    if (call->callee == "tensor") {
      return DType::kTensor;
    }
    // Unknown function; assume dynamic return.
    return std::nullopt;
  }
  return std::nullopt;
}

void TypeChecker::CheckBlock(const parser::BlockStatement* block) {
  EnterScope();
  for (const auto& stmt : block->statements) {
    CheckStatement(stmt.get());
  }
  ExitScope();
}

void TypeChecker::CheckFunction(parser::FunctionStatement* fn) {
  FunSig sig;
  sig.params.reserve(fn->parameter_types.size());
  for (const auto& ann : fn->parameter_types) {
    if (ann.type && ann.type->dtype) {
      sig.params.push_back(ann.type->dtype);
    } else {
      sig.params.push_back(std::nullopt);  // dynamic parameter
    }
  }
  if (fn->return_type.type && fn->return_type.type->dtype) {
    sig.ret = fn->return_type.type->dtype;
  }
  functions_[fn->name] = sig;

  EnterScope();
  for (size_t i = 0; i < fn->parameters.size(); ++i) {
    scopes_.back()[fn->parameters[i]] = sig.params[i];
  }
  auto saved_ret = expected_return_;
  expected_return_ = sig.ret;
  CheckStatement(fn->body.get());
  expected_return_ = saved_ret;
  ExitScope();
}

void TypeChecker::CheckStatement(parser::Statement* stmt) {
  using parser::AssignmentStatement;
  using parser::BlockStatement;
  using parser::BreakStatement;
  using parser::ContinueStatement;
  using parser::ExpressionStatement;
  using parser::ForStatement;
  using parser::FunctionStatement;
  using parser::IfStatement;
  using parser::ReturnStatement;
  using parser::WhileStatement;

  if (auto* block = dynamic_cast<BlockStatement*>(stmt)) {
    CheckBlock(block);
    return;
  }
  if (auto* ifs = dynamic_cast<IfStatement*>(stmt)) {
    auto cond_t = TypeOf(ifs->condition.get());
    if (cond_t.has_value() && cond_t.value() != DType::kBool) {
      throw util::Error("Condition must be bool", 0, 0);
    }
    CheckStatement(ifs->then_branch.get());
    if (ifs->else_branch) CheckStatement(ifs->else_branch.get());
    return;
  }
  if (auto* ws = dynamic_cast<WhileStatement*>(stmt)) {
    auto cond_t = TypeOf(ws->condition.get());
    if (cond_t.has_value() && cond_t.value() != DType::kBool) {
      throw util::Error("Condition must be bool", 0, 0);
    }
    CheckStatement(ws->body.get());
    return;
  }
  if (auto* fs = dynamic_cast<ForStatement*>(stmt)) {
    EnterScope();
    if (fs->init) CheckStatement(fs->init.get());
    if (fs->condition) {
      auto cond_t = TypeOf(fs->condition.get());
      if (cond_t.has_value() && cond_t.value() != DType::kBool) {
        throw util::Error("Condition must be bool", 0, 0);
      }
    }
    if (fs->increment) CheckStatement(fs->increment.get());
    if (fs->body) CheckStatement(fs->body.get());
    ExitScope();
    return;
  }
  if (auto* ret = dynamic_cast<ReturnStatement*>(stmt)) {
    if (expected_return_.has_value()) {
      if (ret->expr) {
        auto rt = TypeOf(ret->expr.get());
        if (!IsAssignable(rt, expected_return_)) {
          throw util::Error("Return type mismatch", 0, 0);
        }
      } else {
        throw util::Error("Missing return value", 0, 0);
      }
    }
    return;
  }
  if (auto* fn = dynamic_cast<FunctionStatement*>(stmt)) {
    CheckFunction(fn);
    return;
  }
  if (auto* asn = dynamic_cast<AssignmentStatement*>(stmt)) {
    auto val_t = TypeOf(asn->value.get());
    std::optional<DType> annot;
    if (asn->annotation.type && asn->annotation.type->dtype) {
      annot = asn->annotation.type->dtype;
    }
    if (annot.has_value()) {
      if (!IsAssignable(val_t, annot)) {
        throw util::Error("Type mismatch in assignment to " + asn->name, 0, 0);
      }
      scopes_.back()[asn->name] = annot;
    } else {
      // Unannotated bindings remain dynamic.
      scopes_.back()[asn->name] = std::nullopt;
    }
    return;
  }
  if (dynamic_cast<BreakStatement*>(stmt) || dynamic_cast<ContinueStatement*>(stmt)) {
    return;
  }
  if (auto* expr_stmt = dynamic_cast<ExpressionStatement*>(stmt)) {
    (void)TypeOf(expr_stmt->expr.get());
    return;
  }
}

void TypeChecker::Check(parser::Statement* stmt) {
  CheckStatement(stmt);
}

}  // namespace lattice::runtime
