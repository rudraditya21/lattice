# Grammar and Syntax

Notation: `*` zero or more, `+` one or more, `?` optional, `|` choice, terminals in quotes.

## Lexical Elements
- Identifiers: `[A-Za-z_][A-Za-z0-9_]*`
- Numbers: integers or floats (`123`, `3.14`)
- Keywords: `if`, `else`, `while`, `for`, `break`, `continue`, `func`, `return`, `true`, `false`
- Operators: `+ - * / == != > < >= <= = , : ; ( ) { }`

## Expressions
```
expression    → equality ;
equality      → comparison ( ( "==" | "!=" ) comparison )* ;
comparison    → term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
term          → factor ( ( "+" | "-" ) factor )* ;
factor        → unary ( ( "*" | "/" ) unary )* ;
unary         → "-" unary | primary ;
primary       → NUMBER | "true" | "false" | IDENTIFIER func_call? | "(" expression ")" ;
func_call     → "(" arguments? ")" ;
arguments     → expression ( "," expression )* ;
```

## Statements
```
statement     → if_stmt
              | while_stmt
              | for_stmt
              | func_def
              | return_stmt
              | block
              | assignment_or_expr ;

block         → "{" statement* "}" ;

if_stmt       → "if" "(" expression ")" statement ( "else" statement )? ;

while_stmt    → "while" "(" expression ")" statement ;

for_stmt      → "for" "(" for_init? ";" for_cond? ";" for_incr? ")" statement ;
for_init      → assignment_or_expr ;
for_cond      → expression ;
for_incr      → assignment_or_expr ;

func_def      → "func" IDENTIFIER "(" parameters? ")" return_annot? statement ;
parameters    → IDENTIFIER type_annot? ( "," IDENTIFIER type_annot? )* ;
return_annot  → "->" type_annot ;

return_stmt   → "return" expression? ;

assignment_or_expr → IDENTIFIER type_annot? "=" expression
                   | expression ;

type_annot    → ":" IDENTIFIER ;
```
- Semicolons are optional after statements but required inside `for` headers.
- Blocks do not currently introduce new variable scopes (all bindings are in the current environment).

## Truthiness
- Control flow requires `bool` conditions. Arithmetic still treats `true/false` as `1/0`. Comparisons yield booleans (`true`/`false`).

## Functions
- Defined with `func name(params) { ... }`.
- `return expr;` exits the function; if omitted, the last statement value is returned if present, otherwise `0`.
- Functions capture their defining environment and can call other user or builtin functions (e.g., `print`). Annotations on params/returns are enforced; unannotated code remains dynamic.

## Types
- Type annotations use built-in names: `bool`, `i8/i16/i32/i64/u8/u16/u32/u64`, `f16/bfloat16/f32/f64`, `complex64/complex128`, `decimal`, `rational`, `tensor`.
- Typed constructors/casts: `int()`, `float()`, `complex()`, `decimal()`, `rational()`, `tensor()`.
- Promotion: complex > float > int; signed/unsigned width-aware; decimal↔decimal and rational↔rational only; tensor ops promote element dtype; implicit narrowing is rejected (use casts).
