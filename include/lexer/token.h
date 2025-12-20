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
  kLParen,
  kRParen,
  kInvalid,
};

struct Token {
  TokenType type;
  std::string lexeme;
  int line;
  int column;
};

}  // namespace lattice::lexer

#endif  // LATTICE_LEXER_TOKEN_H_
