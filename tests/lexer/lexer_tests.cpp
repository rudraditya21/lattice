#include <vector>

#include "test_util.h"

namespace test {

void RunLexerTests(TestContext* ctx) {
  lx::Lexer lex("+ - * / ( ) , foo 123");
  std::vector<lx::TokenType> types;
  while (true) {
    auto tok = lex.NextToken();
    types.push_back(tok.type);
    if (tok.type == lx::TokenType::kEof) {
      break;
    }
  }
  std::vector<lx::TokenType> expected = {lx::TokenType::kPlus,   lx::TokenType::kMinus,
                                         lx::TokenType::kStar,   lx::TokenType::kSlash,
                                         lx::TokenType::kLParen, lx::TokenType::kRParen,
                                         lx::TokenType::kComma,  lx::TokenType::kIdentifier,
                                         lx::TokenType::kNumber, lx::TokenType::kEof};
  ExpectTrue(types == expected, "basic_tokenization", ctx);
}

}  // namespace test
