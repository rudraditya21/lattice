#ifndef LATTICE_PARSER_PARSER_H_
#define LATTICE_PARSER_PARSER_H_

#include <memory>

#include "lexer/lexer.h"
#include "parser/ast.h"
#include "util/error.h"

namespace lattice::parser {

class Parser {
 public:
  /// Builds a parser with ownership of the provided lexer.
  explicit Parser(lexer::Lexer lexer);

  /// Parses an expression and throws util::Error on syntax issues.
  std::unique_ptr<Expression> ParseExpression();

  /// Parses a statement (including blocks/conditionals) and throws util::Error on syntax issues.
  std::unique_ptr<Statement> ParseStatement();

 private:
  const lexer::Token& Peek() const;
  const lexer::Token& Next() const;
  const lexer::Token& Previous() const;
  lexer::Token Advance();
  bool Match(lexer::TokenType type);
  void Consume(lexer::TokenType type, const std::string& message);

  std::unique_ptr<Statement> StatementRule();
  std::unique_ptr<Statement> IfStatementRule();
  std::unique_ptr<Statement> Block();
  std::unique_ptr<Statement> AssignmentOrExpression();
  std::unique_ptr<Expression> ExpressionRule();
  std::unique_ptr<Expression> Term();
  std::unique_ptr<Expression> Factor();
  std::unique_ptr<Expression> Unary();
  std::unique_ptr<Expression> Primary();
  std::unique_ptr<Expression> FinishCall(std::string callee);

  lexer::Lexer lexer_;
  lexer::Token current_;
  lexer::Token lookahead_;
  lexer::Token previous_;
};

}  // namespace lattice::parser

#endif  // LATTICE_PARSER_PARSER_H_
