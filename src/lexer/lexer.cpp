#include "lexer/lexer.h"

#include <cctype>

namespace lattice::lexer {

Lexer::Lexer(const std::string& source) : source_(source), index_(0), line_(1), column_(1) {}

char Lexer::Peek() const {
  if (index_ >= source_.size()) {
    return '\0';
  }
  return source_[index_];
}

char Lexer::Advance() {
  char ch = Peek();
  ++index_;
  if (ch == '\n') {
    ++line_;
    column_ = 1;
  } else {
    ++column_;
  }
  return ch;
}

bool Lexer::IsAtEnd() const {
  return index_ >= source_.size();
}

void Lexer::SkipWhitespace() {
  while (!IsAtEnd()) {
    char ch = Peek();
    if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') {
      Advance();
    } else {
      break;
    }
  }
}

Token Lexer::NumberToken() {
  int token_line = line_;
  int token_column = column_;
  std::string lexeme;
  while (std::isdigit(Peek())) {
    lexeme.push_back(Advance());
  }
  if (Peek() == '.') {
    lexeme.push_back(Advance());
    if (!std::isdigit(Peek())) {
      throw util::Error("Invalid number format", token_line, token_column);
    }
    while (std::isdigit(Peek())) {
      lexeme.push_back(Advance());
    }
  }
  return Token{TokenType::kNumber, lexeme, token_line, token_column};
}

Token Lexer::IdentifierToken() {
  int token_line = line_;
  int token_column = column_;
  std::string lexeme;
  while (std::isalnum(Peek()) || Peek() == '_') {
    lexeme.push_back(Advance());
  }
  return Token{TokenType::kIdentifier, lexeme, token_line, token_column};
}

Token Lexer::NextToken() {
  SkipWhitespace();
  int token_line = line_;
  int token_column = column_;

  if (IsAtEnd()) {
    return Token{TokenType::kEof, "", token_line, token_column};
  }

  char ch = Advance();
  switch (ch) {
    case '+':
      return Token{TokenType::kPlus, "+", token_line, token_column};
    case '-':
      return Token{TokenType::kMinus, "-", token_line, token_column};
    case '*':
      return Token{TokenType::kStar, "*", token_line, token_column};
    case '/':
      return Token{TokenType::kSlash, "/", token_line, token_column};
    case ',':
      return Token{TokenType::kComma, ",", token_line, token_column};
    case '=':
      return Token{TokenType::kEqual, "=", token_line, token_column};
    case '(':
      return Token{TokenType::kLParen, "(", token_line, token_column};
    case ')':
      return Token{TokenType::kRParen, ")", token_line, token_column};
    default:
      break;
  }

  if (std::isdigit(ch)) {
    --index_;
    --column_;
    return NumberToken();
  }

  if (std::isalpha(ch) || ch == '_') {
    --index_;
    --column_;
    return IdentifierToken();
  }

  return Token{TokenType::kInvalid, std::string(1, ch), token_line, token_column};
}

}  // namespace lattice::lexer
