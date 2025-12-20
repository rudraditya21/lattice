#ifndef LATTICE_PARSER_AST_H_
#define LATTICE_PARSER_AST_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lattice::parser {

enum class UnaryOp { kNegate };
enum class BinaryOp { kAdd, kSub, kMul, kDiv };

struct Expression {
  virtual ~Expression() = default;
};

/// Numeric literal value.
struct NumberLiteral : public Expression {
  explicit NumberLiteral(double v) : value(v) {}
  double value;
};

/// Unary expression such as negation.
struct UnaryExpression : public Expression {
  UnaryExpression(UnaryOp o, std::unique_ptr<Expression> expr) : op(o), operand(std::move(expr)) {}
  UnaryOp op;
  std::unique_ptr<Expression> operand;
};

/// Binary expression for arithmetic operators.
struct BinaryExpression : public Expression {
  BinaryExpression(BinaryOp o, std::unique_ptr<Expression> lhs_expr,
                   std::unique_ptr<Expression> rhs_expr)
      : op(o), lhs(std::move(lhs_expr)), rhs(std::move(rhs_expr)) {}
  BinaryOp op;
  std::unique_ptr<Expression> lhs;
  std::unique_ptr<Expression> rhs;
};

/// Named identifier reference.
struct Identifier : public Expression {
  explicit Identifier(std::string n) : name(std::move(n)) {}
  std::string name;
};

/// Function call with positional arguments.
struct CallExpression : public Expression {
  CallExpression(std::string callee_name, std::vector<std::unique_ptr<Expression>> arguments)
      : callee(std::move(callee_name)), args(std::move(arguments)) {}
  std::string callee;
  std::vector<std::unique_ptr<Expression>> args;
};

}  // namespace lattice::parser

#endif  // LATTICE_PARSER_AST_H_
