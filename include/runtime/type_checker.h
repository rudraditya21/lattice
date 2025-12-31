#ifndef LATTICE_RUNTIME_TYPE_CHECKER_H_
#define LATTICE_RUNTIME_TYPE_CHECKER_H_

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "parser/ast.h"
#include "runtime/dtype.h"

namespace lattice::runtime {

// A lightweight static type checker that validates annotated parameters/returns/bindings,
// propagates basic expression types, and rejects implicit narrowing. Explicit narrowing is
// allowed via the cast builtin: cast(typeName, expr).
class TypeChecker {
 public:
  TypeChecker();
  void Check(parser::Statement* stmt);

 private:
  struct FunSig {
    std::vector<std::optional<Type>> params;
    std::optional<Type> ret;
  };

  using Scope = std::unordered_map<std::string, std::optional<Type>>;

  std::optional<Type> TypeOf(const parser::Expression* expr);
  void CheckStatement(parser::Statement* stmt);
  void CheckBlock(const parser::BlockStatement* block);
  void CheckFunction(parser::FunctionStatement* fn);
  void BindName(const std::string& name, std::optional<Type> type);
  bool IsAssignable(const std::optional<Type>& from, const std::optional<Type>& to);
  void EnterScope();
  void ExitScope();

  std::vector<Scope> scopes_;
  std::unordered_map<std::string, FunSig> functions_;
  std::optional<Type> expected_return_;
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_TYPE_CHECKER_H_
