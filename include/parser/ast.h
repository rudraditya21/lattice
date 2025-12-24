#ifndef LATTICE_PARSER_AST_H_
#define LATTICE_PARSER_AST_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

// AST nodes for expressions and statements.

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

struct Statement {
  virtual ~Statement() = default;
};

/// Expression used as a statement.
struct ExpressionStatement : public Statement {
  explicit ExpressionStatement(std::unique_ptr<Expression> e) : expr(std::move(e)) {}
  std::unique_ptr<Expression> expr;
};

/// Assignment to a named identifier.
struct AssignmentStatement : public Statement {
  AssignmentStatement(std::string n, std::unique_ptr<Expression> v)
      : name(std::move(n)), value(std::move(v)) {}
  std::string name;
  std::unique_ptr<Expression> value;
};

/// Sequence of statements in a block.
struct BlockStatement : public Statement {
  explicit BlockStatement(std::vector<std::unique_ptr<Statement>> stmts)
      : statements(std::move(stmts)) {}
  std::vector<std::unique_ptr<Statement>> statements;
};

/// Conditional statement with optional else branch.
struct IfStatement : public Statement {
  IfStatement(std::unique_ptr<Expression> cond, std::unique_ptr<Statement> then_branch,
              std::unique_ptr<Statement> else_branch)
      : condition(std::move(cond)),
        then_branch(std::move(then_branch)),
        else_branch(std::move(else_branch)) {}
  std::unique_ptr<Expression> condition;
  std::unique_ptr<Statement> then_branch;
  std::unique_ptr<Statement> else_branch;
};

}  // namespace lattice::parser

#endif  // LATTICE_PARSER_AST_H_
