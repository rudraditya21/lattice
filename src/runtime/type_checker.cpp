#include "runtime/type_checker.h"

#include <numeric>

#include "runtime/ops.h"
#include "util/error.h"

namespace lattice::runtime {

namespace {

std::optional<Type> LookupDType(const std::string& name) {
  if (name == "bool") return Type{DType::kBool};
  if (name == "i8") return Type{DType::kI8};
  if (name == "i16") return Type{DType::kI16};
  if (name == "i32") return Type{DType::kI32};
  if (name == "i64") return Type{DType::kI64};
  if (name == "u8") return Type{DType::kU8};
  if (name == "u16") return Type{DType::kU16};
  if (name == "u32") return Type{DType::kU32};
  if (name == "u64") return Type{DType::kU64};
  if (name == "f16") return Type{DType::kF16};
  if (name == "bfloat16") return Type{DType::kBF16};
  if (name == "f32") return Type{DType::kF32};
  if (name == "f64") return Type{DType::kF64};
  if (name == "complex64") return Type{DType::kC64};
  if (name == "complex128") return Type{DType::kC128};
  if (name == "string") return Type{DType::kString};
  if (name == "decimal") return Type{DType::kDecimal};
  if (name == "rational") return Type{DType::kRational};
  if (name == "tensor") return Type{DType::kTensor};
  return std::nullopt;
}

std::string DTypeName(DType t) {
  switch (t) {
    case DType::kBool:
      return "bool";
    case DType::kI8:
      return "i8";
    case DType::kI16:
      return "i16";
    case DType::kI32:
      return "i32";
    case DType::kI64:
      return "i64";
    case DType::kU8:
      return "u8";
    case DType::kU16:
      return "u16";
    case DType::kU32:
      return "u32";
    case DType::kU64:
      return "u64";
    case DType::kF16:
      return "f16";
    case DType::kBF16:
      return "bfloat16";
    case DType::kF32:
      return "f32";
    case DType::kF64:
      return "f64";
    case DType::kC64:
      return "complex64";
    case DType::kC128:
      return "complex128";
    case DType::kString:
      return "string";
    case DType::kDecimal:
      return "decimal";
    case DType::kRational:
      return "rational";
    case DType::kFunction:
      return "function";
    case DType::kTensor:
      return "tensor";
    case DType::kTuple:
      return "tuple";
    case DType::kRecord:
      return "record";
  }
  return "unknown";
}

std::string TypeNameStr(const Type& t) {
  if (t.kind == DType::kTuple) {
    std::string s = "(";
    for (size_t i = 0; i < t.tuple_elems.size(); ++i) {
      if (i > 0) s += ", ";
      s += t.tuple_elems[i].has_value() ? DTypeName(t.tuple_elems[i].value()) : "dynamic";
    }
    s += ")";
    return s;
  }
  if (t.kind == DType::kRecord) {
    std::string s = "{";
    for (size_t i = 0; i < t.record_fields.size(); ++i) {
      if (i > 0) s += ", ";
      s += t.record_fields[i].first + ": " +
           (t.record_fields[i].second.has_value() ? DTypeName(t.record_fields[i].second.value())
                                                  : "dynamic");
    }
    s += "}";
    return s;
  }
  return DTypeName(t.kind);
}

std::string OptTypeName(const std::optional<Type>& t) {
  return t.has_value() ? TypeNameStr(t.value()) : "dynamic";
}

bool IsAssignableD(const std::optional<DType>& from, const std::optional<DType>& to) {
  if (!to.has_value()) return true;
  if (!from.has_value()) return true;
  if (from.value() == to.value()) return true;
  return PromoteType(from.value(), to.value()) == to.value();
}

bool IsAssignableType(const Type& from, const Type& to) {
  if (to.kind == DType::kTuple && from.kind == DType::kTuple) {
    if (to.tuple_elems.size() != from.tuple_elems.size()) return false;
    for (size_t i = 0; i < to.tuple_elems.size(); ++i) {
      if (!IsAssignableD(from.tuple_elems[i], to.tuple_elems[i])) return false;
    }
    return true;
  }
  if (to.kind == DType::kRecord && from.kind == DType::kRecord) {
    if (to.record_fields.size() != from.record_fields.size()) return false;
    for (size_t i = 0; i < to.record_fields.size(); ++i) {
      if (to.record_fields[i].first != from.record_fields[i].first) return false;
      if (!IsAssignableD(from.record_fields[i].second, to.record_fields[i].second)) return false;
    }
    return true;
  }
  if (to.kind == DType::kTensor) return from.kind == DType::kTensor;
  if (from.kind == to.kind) return true;
  return PromoteType(from.kind, to.kind) == to.kind;
}

}  // namespace

TypeChecker::TypeChecker() {
  EnterScope();
  // Minimal builtin signatures to allow checking common calls.
  functions_["print"] = FunSig{{std::nullopt}, std::nullopt};
  functions_["pow"] = FunSig{{std::nullopt, std::nullopt}, Type{DType::kF64}};
  functions_["gcd"] = FunSig{{std::nullopt, std::nullopt}, Type{DType::kI64}};
  functions_["lcm"] = FunSig{{std::nullopt, std::nullopt}, Type{DType::kI64}};
  functions_["abs"] = FunSig{{std::nullopt}, std::nullopt};
  functions_["set_decimal_precision"] = FunSig{{Type{DType::kI32}}, std::nullopt};
  functions_["get_decimal_precision"] = FunSig{{}, Type{DType::kI32}};
  functions_["int"] = FunSig{{std::nullopt}, Type{DType::kI64}};
  functions_["float"] = FunSig{{std::nullopt}, Type{DType::kF64}};
  functions_["decimal"] = FunSig{{std::nullopt}, Type{DType::kDecimal}};
  functions_["rational"] = FunSig{{std::nullopt, std::nullopt}, Type{DType::kRational}};
  functions_["complex"] = FunSig{{std::nullopt, std::nullopt}, Type{DType::kC128}};
  functions_["tensor"] = FunSig{{std::nullopt, std::nullopt}, Type{DType::kTensor}};
  functions_["tensor_values"] = FunSig{{std::nullopt}, Type{DType::kTensor}};
  functions_["sum"] = FunSig{{Type{DType::kTensor}}, Type{DType::kF64}};
  functions_["mean"] = FunSig{{Type{DType::kTensor}}, Type{DType::kF64}};
  functions_["len"] = FunSig{{std::nullopt}, Type{DType::kI64}};
  functions_["keys"] = FunSig{{Type{DType::kRecord}}, Type{DType::kTuple}};
  functions_["values"] = FunSig{{Type{DType::kRecord}}, Type{DType::kTuple}};
  functions_["has_key"] = FunSig{{Type{DType::kRecord}, Type{DType::kString}}, Type{DType::kBool}};
}

void TypeChecker::EnterScope() {
  scopes_.emplace_back();
}

void TypeChecker::ExitScope() {
  scopes_.pop_back();
}

bool TypeChecker::IsAssignable(const std::optional<Type>& from, const std::optional<Type>& to) {
  if (!to.has_value()) return true;
  if (!from.has_value()) return true;
  return IsAssignableType(from.value(), to.value());
}

std::optional<Type> TypeChecker::TypeOf(const parser::Expression* expr) {
  using parser::BinaryExpression;
  using parser::BinaryOp;
  using parser::BoolLiteral;
  using parser::CallExpression;
  using parser::Identifier;
  using parser::IndexExpression;
  using parser::NumberLiteral;
  using parser::RecordLiteral;
  using parser::StringLiteral;
  using parser::TupleLiteral;
  using parser::UnaryExpression;
  using parser::UnaryOp;

  if (auto num = dynamic_cast<const NumberLiteral*>(expr)) {
    if (num->is_integer_token) return Type{DType::kI32};
    size_t digits = 0;
    for (char c : num->lexeme) {
      if (std::isdigit(static_cast<unsigned char>(c))) ++digits;
    }
    return digits <= 7 ? Type{DType::kF32} : Type{DType::kF64};
  }
  if (dynamic_cast<const BoolLiteral*>(expr)) {
    return Type{DType::kBool};
  }
  if (dynamic_cast<const StringLiteral*>(expr)) {
    return Type{DType::kString};
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
      return Type{DType::kBool};
    }
    return Type{PromoteType(lt->kind, rt->kind)};
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
            throw util::Error("Type mismatch for argument " + std::to_string(i + 1) + " to " +
                                  call->callee + " (expected " + OptTypeName(sig.params[i]) +
                                  ", got " + OptTypeName(arg_t) + ")",
                              0, 0);
          }
        }
      }
      return sig.ret;
    }
    if (call->callee == "tensor") {
      return Type{DType::kTensor};
    }
    return std::nullopt;
  }
  if (auto tuple = dynamic_cast<const TupleLiteral*>(expr)) {
    Type t;
    t.kind = DType::kTuple;
    for (const auto& e : tuple->elements) {
      auto et = TypeOf(e.get());
      t.tuple_elems.push_back(et ? std::optional<DType>(et->kind) : std::nullopt);
    }
    return t;
  }
  if (auto record = dynamic_cast<const RecordLiteral*>(expr)) {
    Type t;
    t.kind = DType::kRecord;
    for (const auto& f : record->fields) {
      auto ft = TypeOf(f.second.get());
      t.record_fields.emplace_back(f.first, ft ? std::optional<DType>(ft->kind) : std::nullopt);
    }
    return t;
  }
  if (auto idx = dynamic_cast<const IndexExpression*>(expr)) {
    auto obj_t = TypeOf(idx->object.get());
    if (!obj_t.has_value()) return std::nullopt;
    if (obj_t->kind == DType::kTuple) {
      if (auto num = dynamic_cast<const NumberLiteral*>(idx->index.get())) {
        int64_t pos = static_cast<int64_t>(num->value);
        if (pos >= 0 && pos < static_cast<int64_t>(obj_t->tuple_elems.size())) {
          return obj_t->tuple_elems[static_cast<size_t>(pos)]
                     ? std::optional<Type>{Type{
                           obj_t->tuple_elems[static_cast<size_t>(pos)].value()}}
                     : std::nullopt;
        }
      }
      return std::nullopt;
    }
    if (obj_t->kind == DType::kRecord) {
      if (auto key = dynamic_cast<const StringLiteral*>(idx->index.get())) {
        for (const auto& f : obj_t->record_fields) {
          if (f.first == key->value) {
            return f.second.has_value() ? std::optional<Type>{Type{f.second.value()}}
                                        : std::nullopt;
          }
        }
        throw util::Error("Record key not found: " + key->value, 0, 0);
      }
      return std::nullopt;
    }
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
      sig.params.push_back(Type{ann.type->dtype.value()});
    } else {
      sig.params.push_back(std::nullopt);
    }
  }
  if (fn->return_type.type && fn->return_type.type->dtype) {
    sig.ret = Type{fn->return_type.type->dtype.value()};
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
    if (cond_t.has_value() && cond_t->kind != DType::kBool) {
      throw util::Error("Condition must be bool", 0, 0);
    }
    CheckStatement(ifs->then_branch.get());
    if (ifs->else_branch) CheckStatement(ifs->else_branch.get());
    return;
  }
  if (auto* ws = dynamic_cast<WhileStatement*>(stmt)) {
    auto cond_t = TypeOf(ws->condition.get());
    if (cond_t.has_value() && cond_t->kind != DType::kBool) {
      throw util::Error("Condition must be bool (got " + OptTypeName(cond_t) + ")", 0, 0);
    }
    CheckStatement(ws->body.get());
    return;
  }
  if (auto* fs = dynamic_cast<ForStatement*>(stmt)) {
    EnterScope();
    if (fs->init) CheckStatement(fs->init.get());
    if (fs->condition) {
      auto cond_t = TypeOf(fs->condition.get());
      if (cond_t.has_value() && cond_t->kind != DType::kBool) {
        throw util::Error("Condition must be bool (got " + OptTypeName(cond_t) + ")", 0, 0);
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
          throw util::Error("Return type mismatch (expected " + OptTypeName(expected_return_) +
                                ", got " + OptTypeName(rt) + ")",
                            0, 0);
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
    if (asn->tuple_pattern) {
      if (!val_t.has_value() || val_t->kind != DType::kTuple) {
        throw util::Error("Tuple destructuring expects a tuple value", asn->tuple_pattern->line,
                          asn->tuple_pattern->column);
      }
      if (val_t->tuple_elems.size() != asn->tuple_pattern->names.size()) {
        throw util::Error("Tuple destructuring arity mismatch", asn->tuple_pattern->line,
                          asn->tuple_pattern->column);
      }
      for (size_t i = 0; i < asn->tuple_pattern->names.size(); ++i) {
        scopes_.back()[asn->tuple_pattern->names[i]] =
            val_t->tuple_elems[i].has_value()
                ? std::optional<Type>{Type{val_t->tuple_elems[i].value()}}
                : std::nullopt;
      }
      return;
    }
    if (asn->record_pattern) {
      if (!val_t.has_value() || val_t->kind != DType::kRecord) {
        throw util::Error("Record destructuring expects a record value", asn->record_pattern->line,
                          asn->record_pattern->column);
      }
      if (val_t->record_fields.size() != asn->record_pattern->fields.size()) {
        throw util::Error("Record destructuring field mismatch", asn->record_pattern->line,
                          asn->record_pattern->column);
      }
      for (const auto& field : asn->record_pattern->fields) {
        bool found = false;
        for (const auto& tfield : val_t->record_fields) {
          if (tfield.first == field.first) {
            found = true;
            scopes_.back()[field.second] = tfield.second.has_value()
                                               ? std::optional<Type>{Type{tfield.second.value()}}
                                               : std::nullopt;
            break;
          }
        }
        if (!found) {
          throw util::Error("Record key not found in destructuring: " + field.first,
                            asn->record_pattern->line, asn->record_pattern->column);
        }
      }
      return;
    }
    std::optional<Type> annot;
    if (asn->annotation.type && asn->annotation.type->dtype) {
      annot = Type{asn->annotation.type->dtype.value()};
    }
    if (annot.has_value()) {
      if (!IsAssignable(val_t, annot)) {
        throw util::Error("Type mismatch in assignment to " + asn->name + " (expected " +
                              OptTypeName(annot) + ", got " + OptTypeName(val_t) + ")",
                          0, 0);
      }
      scopes_.back()[asn->name] = annot;
    } else {
      scopes_.back()[asn->name] = val_t;
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
