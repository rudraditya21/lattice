# Lattice Semantic Spec (Draft)

Note: This document is the source of truth for scoping and control-flow truthiness. The grammar
document describes syntax only.

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
- Simple assignment rebinds the nearest existing binding; if the name is unbound in all enclosing
  scopes, it creates a new binding in the current scope.
- Assignment is a statement, not an expression; `x = y = 3` is not permitted.
- Annotated bindings (`x: i32 = 3`) enforce the declared type; unannotated remain dynamic.
- Annotations apply to simple bindings only; destructuring assignments do not support annotations.
- Function parameters are immutable bindings inside the function body unless explicitly reassigned.

## Evaluation Model
- Lexical scoping; blocks `{ ... }` introduce new scopes for bindings, and `for` headers introduce a
  loop-local scope shared with the loop body. Lookup resolves to the nearest enclosing binding.
- Call-by-value: arguments are evaluated left-to-right before a call; the callee body executes after
  all arguments are computed.
- Deterministic evaluation order: subexpressions evaluate left-to-right; there is no reordering of
  side effects or control-flow guards.
- Control flow: `if/else`, `while`, `for` with `break`/`continue`; `return` exits the current
  function. Conditions must evaluate to `bool`.
- Aggregates:
  - Tuples: immutable, positional, created with `(a, b)` (singleton `(a,)`). Indexing `t[0]` bounds-checks. Equality is structural. Tuples are the only literal collection.
  - Records: immutable, ordered fields `{x: a, y: b}`; keys are identifiers or strings. Access via `r["x"]`; missing keys are errors. Equality is structural (field names/order and values). Square-bracket list literals are not supported.
  - Strings: immutable; only `==/!=` supported.

## Tensors and Broadcasting
- Dense, row-major storage with explicit shape metadata; dimensions must be positive.
- Elementwise ops (`+ - * /`) follow NumPy-style broadcasting: align trailing dimensions; a
  dimension may be `1` or equal to the other operand; scalar `()` broadcasts to any shape. Any other
  mismatch raises a runtime error with the expression location.
- Reductions (`sum`, `mean`, `var`, `std`) reduce all elements; dtype promotion follows scalar
  promotion rules; results are cast back to the element dtype. Sparse reductions treat missing
  entries as zero; ragged reductions operate over the flattened values buffer.
- Tensor literals use `tensor(...)` and `tensor_values(...)`; use nested tuples for multi-dimensional
  data (e.g., `tensor_values(((1,2),(3,4)))`). Square-bracket list literals are not supported. Creation
  errors (empty shape, bad dims) report source locations.
- Sparse and ragged: `tensor_sparse_csr`, `tensor_sparse_coo`, `tensor_ragged`, and conversions
  `to_dense`, `to_sparse_csr`, `to_sparse_coo`. Ragged elementwise ops require matching
  `row_splits`; sparse formats must match for sparse⊕sparse; dense⊕sparse densifies sparse.
- Linear algebra: `matmul` and `transpose` support 2D dense tensors (sparse inputs are densified).
  Other LA ops (solve/QR/LU/SVD) are not implemented yet.
- Convolution/pooling: `conv2d` (valid padding, dense only) and `max_pool2d` (integer kernels, no
  stride/dilation yet) operate on 2D tensors.
- FFT: `fft1d` (dense-only) returns `(real_tensor, imag_tensor)`; implementation is naive O(n^2).
- Comparisons on tensors are not supported; elementwise results are scalars/tensors depending on
  operands.

## Error Model
- Errors are fatal for the current execution: evaluation stops and reports `Error at line:col - message`.
- Type errors: mismatched annotations, illegal promotions, or invalid operations on types.
- Runtime errors: division by zero, shape mismatches, out-of-domain math (as applicable).
- Parser/lexer errors: unexpected tokens, unterminated constructs, or invalid literals.
- Determinism: given the same inputs and seed, behavior and errors are stable across runs/platforms.
- NaN/Inf policy: numeric operations (scalars, complex, tensors) must produce finite values. Division
  by zero or any non-finite result raises a runtime error with the source location of the
  expression.
