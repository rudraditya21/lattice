#include "parser/parser.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

namespace lattice::parser {

Parser::Parser(lexer::Lexer lexer)
    : lexer_(std::move(lexer)),
      current_(lexer_.NextToken()),
      previous_{lexer::TokenType::kInvalid, "", 0, 0} {}

const lexer::Token& Parser::Peek() const {
  return current_;
}

const lexer::Token& Parser::Previous() const {
  return previous_;
}

lexer::Token Parser::Advance() {
  previous_ = current_;
  current_ = lexer_.NextToken();
  return previous_;
}

bool Parser::Match(lexer::TokenType type) {
  if (Peek().type == type) {
    Advance();
    return true;
  }
  return false;
}

void Parser::Consume(lexer::TokenType type, const std::string& message) {
  if (Peek().type == type) {
    Advance();
    return;
  }
  throw util::Error(message, Peek().line, Peek().column);
}

std::unique_ptr<Expression> Parser::ParseExpression() {
  auto expr = ExpressionRule();
  if (Peek().type != lexer::TokenType::kEof) {
    throw util::Error("Unexpected tokens after expression", Peek().line, Peek().column);
  }
  return expr;
}

std::unique_ptr<Expression> Parser::ExpressionRule() {
  return Term();
}

std::unique_ptr<Expression> Parser::Term() {
  auto expr = Factor();
  while (true) {
    if (Match(lexer::TokenType::kPlus)) {
      auto rhs = Factor();
      expr = std::make_unique<BinaryExpression>(BinaryOp::kAdd, std::move(expr), std::move(rhs));
      continue;
    }
    if (Match(lexer::TokenType::kMinus)) {
      auto rhs = Factor();
      expr = std::make_unique<BinaryExpression>(BinaryOp::kSub, std::move(expr), std::move(rhs));
      continue;
    }
    break;
  }
  return expr;
}

std::unique_ptr<Expression> Parser::Factor() {
  auto expr = Unary();
  while (true) {
    if (Match(lexer::TokenType::kStar)) {
      auto rhs = Unary();
      expr = std::make_unique<BinaryExpression>(BinaryOp::kMul, std::move(expr), std::move(rhs));
      continue;
    }
    if (Match(lexer::TokenType::kSlash)) {
      auto rhs = Unary();
      expr = std::make_unique<BinaryExpression>(BinaryOp::kDiv, std::move(expr), std::move(rhs));
      continue;
    }
    break;
  }
  return expr;
}

std::unique_ptr<Expression> Parser::Unary() {
  if (Match(lexer::TokenType::kMinus)) {
    auto operand = Unary();
    return std::make_unique<UnaryExpression>(UnaryOp::kNegate, std::move(operand));
  }

  return Primary();
}

std::unique_ptr<Expression> Parser::Primary() {
  if (Match(lexer::TokenType::kNumber)) {
    const auto& tok = Previous();
    double value = std::strtod(tok.lexeme.c_str(), nullptr);
    return std::make_unique<NumberLiteral>(value);
  }

  if (Match(lexer::TokenType::kIdentifier)) {
    const auto& tok = Previous();
    return std::make_unique<Identifier>(tok.lexeme);
  }

  if (Match(lexer::TokenType::kLParen)) {
    auto expr = ExpressionRule();
    Consume(lexer::TokenType::kRParen, "Expected ')'");
    return expr;
  }

  throw util::Error("Unexpected token: " + Peek().lexeme, Peek().line, Peek().column);
}

}  // namespace lattice::parser
