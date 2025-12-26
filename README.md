# lattice

Lattice is a small scientific computing language aimed at reproducible numerics and statistical workflows. It ships a REPL and embeddable library with:
- Arithmetic expressions with identifiers, unary minus, calls, assignments, and blocks.
- Control flow via `if/else` statements, nested blocks, `while` loops, `for` loops, and `break`/`continue`.
- Strict typing with optional annotations and a numeric tower (ints, floats, complex, decimals, rationals) plus tensors.
 - Built-in constants `pi`, `e`, `gamma`, `inf` and math helpers `pow`, `gcd`, `lcm`, `abs`, `sign`, `mod`, `floor`, `ceil`, `round`, `clamp`, `min`, `max`, `sum`, `mean`, `var`, `std`, `transpose`, `matmul`, `conv2d`, `max_pool2d`, `fft1d`.
 - Typed constructors/casts: `int()`, `float()`, `complex()`, `decimal()`, `rational()`, `tensor()`, `tensor_values()`, `tensor_sparse_csr()`, `tensor_sparse_coo()`, `tensor_ragged()`, `to_dense()`, `to_sparse_csr()`, `to_sparse_coo()`.
The project is organized with headers in `include/` and sources in `src/`.

## Layout
- `include/` – public headers (`lexer/`, `parser/`, `runtime/`, `repl/`, `builtin/`, `util/`)
- `src/` – implementation (`lexer/`, `parser/`, `runtime/`, `repl/`, `builtin/`, `main.cpp`)
- `tests/` – modular tests under `tests/{lexer,parser,runtime,builtin,util,repl}` with shared helpers at `tests/test_util.*` and entry in `tests/main.cpp`
- `CMakeLists.txt` – build configuration
- `.clang-format` – formatting rules
- `.gitignore` – repository hygiene

## Build
```bash
cmake -S . -B build
cmake --build build
```

## Run REPL
```bash
./build/lattice
```
Example session (typed values, complex/tensors):
```
lattice> x = 3
3
lattice> x + 2
5
lattice> complex(1, -2)
1+-2i
lattice> t = tensor(2, 2, 1)
tensor[dense][2x2]<f64>
lattice> tensor_values((1,2,3))
tensor[dense][3]<i32>
lattice> tensor_values(((1,2),(3,4)))
tensor[dense][2x2]<i32>
lattice> matmul(tensor_values(((1,2),(3,4))), tensor_values(((1,),(1,))))
tensor[dense][2x1]<f64>
lattice> var(tensor_values((1,3)))
1
lattice> fft1d(tensor_values((1,0,1,0)))
(tensor[dense][4]<f64>, tensor[dense][4]<f64>)
lattice> exit
```

## Tensor Ops and Shapes
- Creation:
  - Dense: `tensor(d0, d1, ..., fill)` for row-major dense; `tensor_values((1,2,3))` for 1D; nested tuples for nD (e.g., `tensor_values(((1,2),(3,4)))`). Square-bracket list literals are **not** supported yet.
  - Sparse: `tensor_sparse_csr((rows, cols), indptr_tuple, indices_tuple, values_tuple)`, `tensor_sparse_coo((rows, cols), rows_tuple, cols_tuple, values_tuple)`.
  - Ragged: `tensor_ragged(row_splits_tuple, values_tuple)`.
  - Conversion: `to_dense`, `to_sparse_csr`, `to_sparse_coo`.
- Elementwise ops: `+ - * /` support dense⊕dense (broadcast), dense⊕sparse (densifies sparse), sparse⊕sparse (same format/shape), ragged⊕ragged (matching `row_splits`). Other mixes error with guidance.
- Reductions: `sum`, `mean`, `var`, `std` reduce all elements; missing sparse entries count as zero; ragged reduces flat values. Results are cast to the tensor element dtype.
- Linear algebra: `matmul` (2D only) and `transpose` (2D only) on dense tensors; sparse inputs are densified first. More ops (solve/QR/LU/SVD) are not implemented yet.
- Convolution/pooling: `conv2d(input2d, kernel2d)` with valid padding only; `max_pool2d(input2d, kh, kw)` with integer kernel sizes, no stride/dilation options yet.
- FFT: `fft1d` returns a tuple `(real_tensor, imag_tensor)`; implementation is naive O(n^2) and dense-only.

## Control Flow
- `if/else`: `if (condition) { expr_or_stmt } else { other_stmt }`. Conditions must be `bool`.
- Comparisons: `==`, `!=`, `>`, `<`, `>=`, `<=` yield booleans (`true`/`false`).
- Boolean literals: `true`, `false`; arithmetic on bools is allowed (1/0) but control-flow requires `bool`.
- `while`: `while (condition) body`. Condition re-evaluated each iteration.
- `for`: `for (init; condition; increment) body`. Any of the three clauses may be empty (e.g., `for (; cond; )`).
- `break` / `continue`: only valid inside loops; `continue` skips to the next iteration; `break` exits the nearest loop. At the REPL top level they print an error.
- Blocks: `{ stmt1; stmt2; ... }` with optional semicolons after statements.
- Functions: `func name(param1, param2) { ... }` defines a function; `return expr;` exits with a value. Without `return`, the last statement value is used if present, otherwise `0`. Functions may carry type annotations on params/returns; annotated boundaries are enforced, unannotated code remains dynamic.

## Types and Promotion
- Numeric tower: `i8/i16/i32/i64/u8/u16/u32/u64`, `f16/bfloat16/f32/f64`, `complex64/complex128`, `decimal`, `rational`, `bool`.
- Aggregates: `tensor` (dense/sparse CSR/COO/ragged) with shape/row-major strides and element dtype (default `f64`), tuples, records, strings.
- Constructors/casts: `int`, `float`, `complex`, `decimal`, `rational`, `tensor`, `tensor_values`, sparse/ragged tensor constructors, `to_dense`, `to_sparse_*`.
- Promotion: complex > float > int; signed/unsigned width-aware; decimal↔decimal and rational↔rational only; tensor ops promote element dtype; implicit narrowing is rejected (use `cast`/constructors).
- Math builtins enforce dtype expectations (e.g., `gcd/lcm` require integers, `abs` handles complex/decimal/rational, `pow` supports complex).

## Tests
```bash
cmake --build build
ctest --test-dir build
```
Tests exercise parsing and evaluation (arithmetic, identifiers, constants, builtins, and error paths).

## Formatting
Use clang-format with the provided configuration:
```bash
find include src \( -name '*.h' -o -name '*.cpp' \) -print0 | xargs -0 clang-format -i
```
