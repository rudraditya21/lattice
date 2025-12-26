# Lattice Semantic Spec (Draft)

## Lexical Grammar (high level)
- Whitespace: spaces/tabs/newlines separate tokens; `//` line comments are skipped.
- Identifiers: `[A-Za-z_][A-Za-z0-9_]*`.
- Literals: integers (decimal), floats (with decimal point or exponent), booleans `true`/`false`, strings (`"text"`).
- Delimiters: `()`, `{}`, `[]`, `,`, `;`, `:`. Operators: `+ - * / == != > >= < <= =`.
- Keywords: `if`, `else`, `while`, `for`, `break`, `continue`, `func`, `return`, type names when used as annotations.

## Expression Precedence (high to low, left-associative unless noted)
1. Parentheses `(expr)`.
2. Unary: `-x`.
3. Multiplicative: `* /`.
4. Additive: `+ -`.
5. Comparisons: `< <= > >=`.
6. Equality: `== !=`.
7. Indexing binds tighter than arithmetic: `postfix [ index ]`.
8. Assignment (right-associative): `=` in statements, not an expression operator in arithmetic contexts.

## Assignment Semantics
- Simple assignment binds or rebinds in the nearest enclosing lexical scope; shadowing is allowed on first introduction.
- Assignment is a statement, not an expression; `x = y = 3` is not permitted.
- Annotated bindings (`x: i32 = 3`) enforce the declared type; unannotated remain dynamic.
- Function parameters are immutable bindings inside the function body unless explicitly reassigned.

## Evaluation Model
- Lexical scoping; blocks `{ ... }` introduce new scopes for bindings.
- Call-by-value: arguments are evaluated left-to-right before a call.
- Deterministic execution order: expressions are evaluated left-to-right; no reordering for side effects.
- Control flow: `if/else`, `while`, `for` with `break`/`continue`; `return` exits the current function.
- Aggregates:
  - Tuples: immutable, positional, created with `(a, b)` (singleton `(a,)`). Indexing `t[0]` bounds-checks. Equality is structural.
  - Records: immutable, ordered fields `{x: a, y: b}`; keys are identifiers or strings. Access via `r["x"]`; missing keys are errors. Equality is structural (field names/order and values).
  - Strings: immutable; only `==/!=` supported.

## Error Model
- Errors are fatal for the current execution: evaluation stops and reports `Error at line:col - message`.
- Type errors: mismatched annotations, illegal promotions, or invalid operations on types.
- Runtime errors: division by zero, shape mismatches, out-of-domain math (as applicable).
- Parser/lexer errors: unexpected tokens, unterminated constructs, or invalid literals.
- Determinism: given the same inputs and seed, behavior and errors are stable across runs/platforms.
- NaN/Inf policy: numeric operations (scalars, complex, tensors) must produce finite values. Division
  by zero or any non-finite result raises a runtime error with the source location of the
  expression.
