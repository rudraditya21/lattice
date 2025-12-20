#ifndef LATTICE_LEXER_LEXER_H_
#define LATTICE_LEXER_LEXER_H_

#include <string>
#include <vector>

#include "lexer/token.h"
#include "util/error.h"

namespace lattice::lexer {

class Lexer {
 public:
  explicit Lexer(const std::string& source);

  Token NextToken();

 private:
  char Peek() const;
  char Advance();
  bool IsAtEnd() const;
  void SkipWhitespace();
  Token NumberToken();
  Token IdentifierToken();

  std::string source_;
  size_t index_;
  int line_;
  int column_;
};

}  // namespace lattice::lexer

#endif  // LATTICE_LEXER_LEXER_H_
