#include "parser/parser.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "runtime/dtype.h"

namespace lattice::parser {

namespace {
std::optional<runtime::DType> LookupDType(const std::string& name) {
  using runtime::DType;
  if (name == "bool") return DType::kBool;
  if (name == "i8") return DType::kI8;
  if (name == "i16") return DType::kI16;
  if (name == "i32") return DType::kI32;
  if (name == "i64") return DType::kI64;
  if (name == "u8") return DType::kU8;
  if (name == "u16") return DType::kU16;
  if (name == "u32") return DType::kU32;
  if (name == "u64") return DType::kU64;
  if (name == "f16") return DType::kF16;
  if (name == "bfloat16") return DType::kBF16;
  if (name == "f32") return DType::kF32;
  if (name == "f64") return DType::kF64;
  if (name == "complex64") return DType::kC64;
  if (name == "complex128") return DType::kC128;
  if (name == "decimal") return DType::kDecimal;
  if (name == "rational") return DType::kRational;
  if (name == "tensor") return DType::kTensor;
  return std::nullopt;
}
}  // namespace

Parser::Parser(lexer::Lexer lexer)
    : lexer_(std::move(lexer)),
      current_(lexer_.NextToken()),
      lookahead_(lexer_.NextToken()),
      previous_{lexer::TokenType::kInvalid, "", 0, 0} {}

const lexer::Token& Parser::Peek() const {
  return current_;
}

const lexer::Token& Parser::Next() const {
  return lookahead_;
}

const lexer::Token& Parser::Previous() const {
  return previous_;
}

lexer::Token Parser::Advance() {
  previous_ = current_;
  current_ = lookahead_;
  lookahead_ = lexer_.NextToken();
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

std::unique_ptr<Statement> Parser::ParseStatement() {
  auto stmt = StatementRule();
  Match(lexer::TokenType::kSemicolon);  // optional trailing semicolon
  while (Peek().type != lexer::TokenType::kEof) {
    Advance();
  }
  return stmt;
}

std::unique_ptr<Statement> Parser::StatementRule() {
  if (Match(lexer::TokenType::kIf)) {
    return IfStatementRule();
  }
  if (Match(lexer::TokenType::kWhile)) {
    return WhileStatementRule();
  }
  if (Match(lexer::TokenType::kFor)) {
    return ForStatementRule();
  }
  if (Match(lexer::TokenType::kFunc)) {
    return FunctionStatementRule();
  }
  if (Match(lexer::TokenType::kReturn)) {
    return ReturnStatementRule();
  }
  if (Match(lexer::TokenType::kLBrace)) {
    return Block();
  }
  if (Match(lexer::TokenType::kBreak)) {
    return std::make_unique<BreakStatement>();
  }
  if (Match(lexer::TokenType::kContinue)) {
    return std::make_unique<ContinueStatement>();
  }
  return AssignmentOrExpression();
}

std::unique_ptr<Statement> Parser::IfStatementRule() {
  int line = Previous().line;
  int col = Previous().column;
  Consume(lexer::TokenType::kLParen, "Expected '(' after 'if'");
  auto condition = ExpressionRule();
  Consume(lexer::TokenType::kRParen, "Expected ')' after condition");
  auto then_branch = StatementRule();
  std::unique_ptr<Statement> else_branch;
  if (Match(lexer::TokenType::kElse)) {
    else_branch = StatementRule();
  }
  auto stmt = std::make_unique<IfStatement>(std::move(condition), std::move(then_branch),
                                            std::move(else_branch));
  stmt->line = line;
  stmt->column = col;
  return stmt;
}

std::unique_ptr<Statement> Parser::WhileStatementRule() {
  int line = Previous().line;
  int col = Previous().column;
  Consume(lexer::TokenType::kLParen, "Expected '(' after 'while'");
  auto condition = ExpressionRule();
  Consume(lexer::TokenType::kRParen, "Expected ')' after condition");
  auto body = StatementRule();
  auto stmt = std::make_unique<WhileStatement>(std::move(condition), std::move(body));
  stmt->line = line;
  stmt->column = col;
  return stmt;
}

std::unique_ptr<Statement> Parser::FunctionStatementRule() {
  int line = Previous().line;
  int col = Previous().column;
  Consume(lexer::TokenType::kIdentifier, "Expected function name");
  std::string name = Previous().lexeme;
  Consume(lexer::TokenType::kLParen, "Expected '(' after function name");
  std::vector<std::string> params;
  std::vector<BindingAnnotation> param_types;
  if (!Match(lexer::TokenType::kRParen)) {
    while (true) {
      Consume(lexer::TokenType::kIdentifier, "Expected parameter name");
      params.push_back(Previous().lexeme);
      BindingAnnotation ann = ParseBindingAnnotation();
      param_types.push_back(std::move(ann));
      if (Match(lexer::TokenType::kRParen)) {
        break;
      }
      Consume(lexer::TokenType::kComma, "Expected ',' between parameters");
    }
  }
  BindingAnnotation return_type;
  if (Peek().type == lexer::TokenType::kMinus && Next().type == lexer::TokenType::kGreater) {
    Advance();  // '-'
    Advance();  // '>'
    if (Peek().type == lexer::TokenType::kIdentifier) {
      auto ret = ParseTypeName();
      return_type = BindingAnnotation(std::move(ret));
    }
  } else if (Match(lexer::TokenType::kColon)) {
    auto ret = ParseTypeName();
    return_type = BindingAnnotation(std::move(ret));
  }
  auto body = StatementRule();
  auto stmt = std::make_unique<FunctionStatement>(std::move(name), std::move(params),
                                                  std::move(param_types), std::move(return_type),
                                                  std::move(body));
  stmt->line = line;
  stmt->column = col;
  return stmt;
}

std::unique_ptr<Statement> Parser::ReturnStatementRule() {
  int line = Previous().line;
  int col = Previous().column;
  if (Peek().type == lexer::TokenType::kSemicolon || Peek().type == lexer::TokenType::kRBrace) {
    auto stmt = std::make_unique<ReturnStatement>(nullptr);
    stmt->line = line;
    stmt->column = col;
    return stmt;
  }
  auto expr = ExpressionRule();
  auto stmt = std::make_unique<ReturnStatement>(std::move(expr));
  stmt->line = line;
  stmt->column = col;
  return stmt;
}

std::unique_ptr<Statement> Parser::ForStatementRule() {
  int line = Previous().line;
  int col = Previous().column;
  Consume(lexer::TokenType::kLParen, "Expected '(' after 'for'");

  std::unique_ptr<Statement> init;
  if (!Match(lexer::TokenType::kSemicolon)) {
    init = AssignmentOrExpression();
    Consume(lexer::TokenType::kSemicolon, "Expected ';' after for-loop initializer");
  }

  std::unique_ptr<Expression> condition;
  if (!Match(lexer::TokenType::kSemicolon)) {
    condition = ExpressionRule();
    Consume(lexer::TokenType::kSemicolon, "Expected ';' after for-loop condition");
  }

  std::unique_ptr<Statement> increment;
  if (Match(lexer::TokenType::kRParen)) {
    // no increment
  } else {
    increment = AssignmentOrExpression();
    Consume(lexer::TokenType::kRParen, "Expected ')' after for-loop increment");
  }

  auto body = StatementRule();
  auto stmt = std::make_unique<ForStatement>(std::move(init), std::move(condition),
                                             std::move(increment), std::move(body));
  stmt->line = line;
  stmt->column = col;
  return stmt;
}

std::unique_ptr<Statement> Parser::Block() {
  std::vector<std::unique_ptr<Statement>> statements;
  while (!Match(lexer::TokenType::kRBrace)) {
    if (Peek().type == lexer::TokenType::kEof) {
      throw util::Error("Unterminated block", Previous().line, Previous().column);
    }
    statements.push_back(StatementRule());
    Match(lexer::TokenType::kSemicolon);
  }
  return std::make_unique<BlockStatement>(std::move(statements));
}

std::unique_ptr<Statement> Parser::AssignmentOrExpression() {
  if (Peek().type == lexer::TokenType::kIdentifier) {
    // Look ahead to see if this is an annotated assignment.
    bool is_assignment = false;
    if (Next().type == lexer::TokenType::kEqual || Next().type == lexer::TokenType::kColon) {
      is_assignment = true;
    }
    if (is_assignment) {
      std::string name = Peek().lexeme;
      int line = Peek().line;
      int col = Peek().column;
      Advance();  // identifier
      BindingAnnotation ann = ParseBindingAnnotation();
      Consume(lexer::TokenType::kEqual, "Expected '=' in assignment");
      auto value = ExpressionRule();
      auto stmt =
          std::make_unique<AssignmentStatement>(std::move(name), std::move(ann), std::move(value));
      stmt->line = line;
      stmt->column = col;
      return stmt;
    }
  }
  if (Peek().type == lexer::TokenType::kLParen) {
    // Tuple destructuring assignment.
    int line = Peek().line;
    int col = Peek().column;
    Advance();  // '('
    std::vector<std::string> names;
    if (!Match(lexer::TokenType::kRParen)) {
      while (true) {
        Consume(lexer::TokenType::kIdentifier, "Expected name in tuple destructuring");
        names.push_back(Previous().lexeme);
        if (Match(lexer::TokenType::kRParen)) break;
        Consume(lexer::TokenType::kComma, "Expected ',' in tuple destructuring");
      }
    }
    auto pattern = std::make_unique<TuplePattern>();
    pattern->names = std::move(names);
    pattern->line = line;
    pattern->column = col;
    BindingAnnotation ann = ParseBindingAnnotation();
    Consume(lexer::TokenType::kEqual, "Expected '=' in destructuring assignment");
    auto value = ExpressionRule();
    auto stmt =
        std::make_unique<AssignmentStatement>(std::move(pattern), std::move(ann), std::move(value));
    stmt->line = line;
    stmt->column = col;
    return stmt;
  }
  if (Peek().type == lexer::TokenType::kLBrace) {
    int line = Peek().line;
    int col = Peek().column;
    Advance();  // '{'
    std::vector<std::pair<std::string, std::string>> fields;
    if (!Match(lexer::TokenType::kRBrace)) {
      while (true) {
        Consume(lexer::TokenType::kIdentifier, "Expected field name in record destructuring");
        std::string key = Previous().lexeme;
        std::string binding = key;
        fields.emplace_back(std::move(key), std::move(binding));
        if (Match(lexer::TokenType::kRBrace)) break;
        Consume(lexer::TokenType::kComma, "Expected ',' in record destructuring");
      }
    }
    auto pattern = std::make_unique<RecordPattern>();
    pattern->fields = std::move(fields);
    pattern->line = line;
    pattern->column = col;
    BindingAnnotation ann = ParseBindingAnnotation();
    Consume(lexer::TokenType::kEqual, "Expected '=' in destructuring assignment");
    auto value = ExpressionRule();
    auto stmt =
        std::make_unique<AssignmentStatement>(std::move(pattern), std::move(ann), std::move(value));
    stmt->line = line;
    stmt->column = col;
    return stmt;
  }
  auto expr = ExpressionRule();
  return std::make_unique<ExpressionStatement>(std::move(expr));
}

std::unique_ptr<Expression> Parser::ExpressionRule() {
  return Equality();
}

std::unique_ptr<Expression> Parser::Equality() {
  auto expr = Comparison();
  while (true) {
    if (Match(lexer::TokenType::kEqualEqual)) {
      auto rhs = Comparison();
      expr = std::make_unique<BinaryExpression>(BinaryOp::kEq, std::move(expr), std::move(rhs));
      continue;
    }
    if (Match(lexer::TokenType::kBangEqual)) {
      auto rhs = Comparison();
      expr = std::make_unique<BinaryExpression>(BinaryOp::kNe, std::move(expr), std::move(rhs));
      continue;
    }
    break;
  }
  return expr;
}

std::unique_ptr<Expression> Parser::Comparison() {
  auto expr = Term();
  while (true) {
    if (Match(lexer::TokenType::kGreater)) {
      auto rhs = Term();
      expr = std::make_unique<BinaryExpression>(BinaryOp::kGt, std::move(expr), std::move(rhs));
      continue;
    }
    if (Match(lexer::TokenType::kGreaterEqual)) {
      auto rhs = Term();
      expr = std::make_unique<BinaryExpression>(BinaryOp::kGe, std::move(expr), std::move(rhs));
      continue;
    }
    if (Match(lexer::TokenType::kLess)) {
      auto rhs = Term();
      expr = std::make_unique<BinaryExpression>(BinaryOp::kLt, std::move(expr), std::move(rhs));
      continue;
    }
    if (Match(lexer::TokenType::kLessEqual)) {
      auto rhs = Term();
      expr = std::make_unique<BinaryExpression>(BinaryOp::kLe, std::move(expr), std::move(rhs));
      continue;
    }
    break;
  }
  return expr;
}

std::unique_ptr<Expression> Parser::Term() {
  auto expr = Factor();
  while (true) {
    if (Match(lexer::TokenType::kPlus)) {
      auto rhs = Factor();
      auto bin =
          std::make_unique<BinaryExpression>(BinaryOp::kAdd, std::move(expr), std::move(rhs));
      bin->line = Previous().line;
      bin->column = Previous().column;
      expr = std::move(bin);
      continue;
    }
    if (Match(lexer::TokenType::kMinus)) {
      auto rhs = Factor();
      auto bin =
          std::make_unique<BinaryExpression>(BinaryOp::kSub, std::move(expr), std::move(rhs));
      bin->line = Previous().line;
      bin->column = Previous().column;
      expr = std::move(bin);
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
      auto bin =
          std::make_unique<BinaryExpression>(BinaryOp::kMul, std::move(expr), std::move(rhs));
      bin->line = Previous().line;
      bin->column = Previous().column;
      expr = std::move(bin);
      continue;
    }
    if (Match(lexer::TokenType::kSlash)) {
      auto rhs = Unary();
      auto bin =
          std::make_unique<BinaryExpression>(BinaryOp::kDiv, std::move(expr), std::move(rhs));
      bin->line = Previous().line;
      bin->column = Previous().column;
      expr = std::move(bin);
      continue;
    }
    break;
  }
  return expr;
}

std::unique_ptr<Expression> Parser::Unary() {
  if (Match(lexer::TokenType::kMinus)) {
    auto operand = Unary();
    auto ue = std::make_unique<UnaryExpression>(UnaryOp::kNegate, std::move(operand));
    ue->line = Previous().line;
    ue->column = Previous().column;
    return ue;
  }

  return Primary();
}

std::unique_ptr<Expression> Parser::Primary() {
  std::unique_ptr<Expression> expr;
  if (Match(lexer::TokenType::kNumber)) {
    const auto& tok = Previous();
    double value = std::strtod(tok.lexeme.c_str(), nullptr);
    bool is_int_token = tok.lexeme.find('.') == std::string::npos &&
                        tok.lexeme.find('e') == std::string::npos &&
                        tok.lexeme.find('E') == std::string::npos;
    expr = std::make_unique<NumberLiteral>(value, is_int_token, tok.lexeme);
    expr->line = tok.line;
    expr->column = tok.column;
  } else if (Match(lexer::TokenType::kString)) {
    expr = std::make_unique<StringLiteral>(Previous().lexeme);
    expr->line = Previous().line;
    expr->column = Previous().column;
  } else if (Match(lexer::TokenType::kTrue)) {
    expr = std::make_unique<BoolLiteral>(true);
    expr->line = Previous().line;
    expr->column = Previous().column;
  } else if (Match(lexer::TokenType::kFalse)) {
    expr = std::make_unique<BoolLiteral>(false);
    expr->line = Previous().line;
    expr->column = Previous().column;
  } else if (Match(lexer::TokenType::kIdentifier)) {
    std::string name = Previous().lexeme;
    if (Match(lexer::TokenType::kLParen)) {
      std::vector<std::unique_ptr<Expression>> args;
      if (!Match(lexer::TokenType::kRParen)) {
        while (true) {
          args.push_back(ExpressionRule());
          if (Match(lexer::TokenType::kRParen)) {
            break;
          }
          Consume(lexer::TokenType::kComma, "Expected ',' between arguments");
        }
      }
      auto call_expr = std::make_unique<CallExpression>(name, std::move(args));
      call_expr->line = Previous().line;
      call_expr->column = Previous().column;
      expr = std::move(call_expr);
    } else {
      expr = std::make_unique<Identifier>(name);
      expr->line = Previous().line;
      expr->column = Previous().column;
    }
  } else if (Match(lexer::TokenType::kLParen)) {
    int l = Previous().line;
    int c = Previous().column;
    if (Match(lexer::TokenType::kRParen)) {
      throw util::Error("Empty tuple not allowed", Previous().line, Previous().column);
    }
    std::vector<std::unique_ptr<Expression>> elems;
    elems.push_back(ExpressionRule());
    if (Match(lexer::TokenType::kRParen)) {
      // Grouping
      expr = std::move(elems[0]);
    } else {
      while (true) {
        if (Match(lexer::TokenType::kRParen)) {
          break;
        }
        Consume(lexer::TokenType::kComma, "Expected ',' in tuple");
        if (Match(lexer::TokenType::kRParen)) {
          // trailing comma after single element (singleton tuple)
          break;
        }
        elems.push_back(ExpressionRule());
      }
      expr = std::make_unique<TupleLiteral>(std::move(elems), l, c);
    }
  } else if (Match(lexer::TokenType::kLBrace)) {
    int l = Previous().line;
    int c = Previous().column;
    std::vector<std::pair<std::string, std::unique_ptr<Expression>>> fields;
    if (!Match(lexer::TokenType::kRBrace)) {
      while (true) {
        if (!(Match(lexer::TokenType::kIdentifier) || Match(lexer::TokenType::kString))) {
          throw util::Error("Expected field name in record", Peek().line, Peek().column);
        }
        std::string key = Previous().lexeme;
        Consume(lexer::TokenType::kColon, "Expected ':' after record field name");
        auto val = ExpressionRule();
        fields.emplace_back(std::move(key), std::move(val));
        if (Match(lexer::TokenType::kRBrace)) {
          break;
        }
        Consume(lexer::TokenType::kComma, "Expected ',' between record fields");
      }
    }
    expr = std::make_unique<RecordLiteral>(std::move(fields), l, c);
  } else {
    throw util::Error("Unexpected token: " + Peek().lexeme, Peek().line, Peek().column);
  }

  // Postfix: indexing
  while (Match(lexer::TokenType::kLBracket)) {
    auto idx = ExpressionRule();
    Consume(lexer::TokenType::kRBracket, "Expected ']' after index");
    expr = std::make_unique<IndexExpression>(std::move(expr), std::move(idx), Previous().line,
                                             Previous().column);
  }
  return expr;
}

std::unique_ptr<TypeName> Parser::ParseTypeName() {
  if (!Match(lexer::TokenType::kIdentifier)) {
    throw util::Error("Expected type name", Peek().line, Peek().column);
  }
  std::string name = Previous().lexeme;
  auto dtype = LookupDType(name);
  if (!dtype.has_value()) {
    throw util::Error("Unknown type: " + name, Previous().line, Previous().column);
  }
  return std::make_unique<TypeName>(std::move(name), dtype);
}

BindingAnnotation Parser::ParseBindingAnnotation() {
  if (Match(lexer::TokenType::kColon)) {
    auto type = ParseTypeName();
    return BindingAnnotation(std::move(type));
  }
  return BindingAnnotation();
}

std::string Parser::TokenTypeName(lexer::TokenType type) const {
  switch (type) {
    case lexer::TokenType::kIdentifier:
      return "identifier";
    default:
      return "token";
  }
}

}  // namespace lattice::parser
