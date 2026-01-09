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

  lx::Lexer string_lex("\"a\\\"b\"");
  auto string_tok = string_lex.NextToken();
  ExpectTrue(string_tok.type == lx::TokenType::kString, "string_token_type", ctx);
  ExpectTrue(string_tok.lexeme == "a\"b", "string_token_lexeme", ctx);

  lx::Lexer comment_lex("foo // comment\nbar");
  auto first = comment_lex.NextToken();
  auto second = comment_lex.NextToken();
  ExpectTrue(first.type == lx::TokenType::kIdentifier, "comment_first_token", ctx);
  ExpectTrue(first.lexeme == "foo", "comment_first_lexeme", ctx);
  ExpectTrue(second.type == lx::TokenType::kIdentifier, "comment_second_token", ctx);
  ExpectTrue(second.lexeme == "bar", "comment_second_lexeme", ctx);

  lx::Lexer invalid_lex("!");
  auto invalid_tok = invalid_lex.NextToken();
  ExpectTrue(invalid_tok.type == lx::TokenType::kInvalid, "invalid_token_type", ctx);

  bool unterminated = false;
  try {
    lx::Lexer bad("\"unterminated");
    bad.NextToken();
  } catch (const util::Error&) {
    unterminated = true;
  }
  ExpectTrue(unterminated, "unterminated_string", ctx);

  bool invalid_number = false;
  try {
    lx::Lexer bad_number("1.");
    bad_number.NextToken();
  } catch (const util::Error&) {
    invalid_number = true;
  }
  ExpectTrue(invalid_number, "invalid_number_format", ctx);
}

}  // namespace test
