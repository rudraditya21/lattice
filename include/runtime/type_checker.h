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
    std::vector<std::optional<DType>> params;
    std::optional<DType> ret;
  };

  using Scope = std::unordered_map<std::string, std::optional<DType>>;

  std::optional<DType> TypeOf(const parser::Expression* expr);
  void CheckStatement(parser::Statement* stmt);
  void CheckBlock(const parser::BlockStatement* block);
  void CheckFunction(parser::FunctionStatement* fn);
  bool IsAssignable(std::optional<DType> from, std::optional<DType> to);
  void EnterScope();
  void ExitScope();

  std::vector<Scope> scopes_;
  std::unordered_map<std::string, FunSig> functions_;
  std::optional<DType> expected_return_;
};

}  // namespace lattice::runtime

#endif  // LATTICE_RUNTIME_TYPE_CHECKER_H_
