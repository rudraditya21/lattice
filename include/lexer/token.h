#ifndef LATTICE_LEXER_TOKEN_H_
#define LATTICE_LEXER_TOKEN_H_

#include <string>

namespace lattice::lexer {

enum class TokenType {
  kEof,
  kNumber,
  kIdentifier,
  kEqual,
  kPlus,
  kMinus,
  kStar,
  kSlash,
  kComma,
  kLParen,
  kRParen,
  kInvalid,
};

/// A lexical token with type, original lexeme, and source location.
struct Token {
  TokenType type;
  std::string lexeme;
  int line;
  int column;
};

}  // namespace lattice::lexer

#endif  // LATTICE_LEXER_TOKEN_H_
