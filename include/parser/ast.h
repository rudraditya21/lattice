#ifndef LATTICE_PARSER_AST_H_
#define LATTICE_PARSER_AST_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

// AST nodes for expressions and statements.

namespace lattice::parser {

enum class UnaryOp { kNegate };
enum class BinaryOp { kAdd, kSub, kMul, kDiv, kEq, kNe, kGt, kGe, kLt, kLe };

struct Expression {
  virtual ~Expression() = default;
};

/// Numeric literal value.
struct NumberLiteral : public Expression {
  NumberLiteral(double v, bool is_int_token, std::string lex)
      : value(v), is_integer_token(is_int_token), lexeme(std::move(lex)) {}
  double value;
  bool is_integer_token;
  std::string lexeme;
};

/// Boolean literal value.
struct BoolLiteral : public Expression {
  explicit BoolLiteral(bool v) : value(v) {}
  bool value;
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

/// Type annotation identifier.
struct TypeName {
  explicit TypeName(std::string n) : name(std::move(n)) {}
  std::string name;
};

/// Function call with positional arguments.
struct CallExpression : public Expression {
  CallExpression(std::string callee_name, std::vector<std::unique_ptr<Expression>> arguments)
      : callee(std::move(callee_name)), args(std::move(arguments)) {}
  std::string callee;
  std::vector<std::unique_ptr<Expression>> args;
};

/// Optional type annotation for bindings.
struct BindingAnnotation {
  BindingAnnotation() = default;
  explicit BindingAnnotation(std::unique_ptr<TypeName> tn) : type(std::move(tn)) {}
  std::unique_ptr<TypeName> type;
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
  AssignmentStatement(std::string n, BindingAnnotation ann, std::unique_ptr<Expression> v)
      : name(std::move(n)), annotation(std::move(ann)), value(std::move(v)) {}
  std::string name;
  BindingAnnotation annotation;
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

/// While loop with a condition and body.
struct WhileStatement : public Statement {
  WhileStatement(std::unique_ptr<Expression> cond, std::unique_ptr<Statement> body_stmt)
      : condition(std::move(cond)), body(std::move(body_stmt)) {}
  std::unique_ptr<Expression> condition;
  std::unique_ptr<Statement> body;
};

/// For loop with optional init/condition/increment parts.
struct ForStatement : public Statement {
  ForStatement(std::unique_ptr<Statement> init_stmt, std::unique_ptr<Expression> cond_expr,
               std::unique_ptr<Statement> incr_stmt, std::unique_ptr<Statement> body_stmt)
      : init(std::move(init_stmt)),
        condition(std::move(cond_expr)),
        increment(std::move(incr_stmt)),
        body(std::move(body_stmt)) {}
  std::unique_ptr<Statement> init;
  std::unique_ptr<Expression> condition;
  std::unique_ptr<Statement> increment;
  std::unique_ptr<Statement> body;
};

/// Break out of the nearest loop.
struct BreakStatement : public Statement {};

/// Continue to the next iteration of the nearest loop.
struct ContinueStatement : public Statement {};

/// Return from the nearest function.
struct ReturnStatement : public Statement {
  explicit ReturnStatement(std::unique_ptr<Expression> e) : expr(std::move(e)) {}
  std::unique_ptr<Expression> expr;
};

/// Function definition statement.
struct FunctionStatement : public Statement {
  FunctionStatement(std::string n, std::vector<std::string> params,
                    std::vector<BindingAnnotation> param_types, BindingAnnotation ret_type,
                    std::unique_ptr<Statement> b)
      : name(std::move(n)),
        parameters(std::move(params)),
        parameter_types(std::move(param_types)),
        return_type(std::move(ret_type)),
        body(std::move(b)) {}
  std::string name;
  std::vector<std::string> parameters;
  std::vector<BindingAnnotation> parameter_types;
  BindingAnnotation return_type;
  std::unique_ptr<Statement> body;
};

}  // namespace lattice::parser

#endif  // LATTICE_PARSER_AST_H_
